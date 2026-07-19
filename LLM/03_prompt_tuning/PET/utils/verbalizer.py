# -*- coding:utf-8 -*-
import sys
from pathlib import Path
from typing import List, Union

for _root in Path(__file__).resolve().parents:
    if (_root / "pet_bootstrap.py").is_file():
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))
        break
else:
    raise RuntimeError("Cannot find PET project root (pet_bootstrap.py)")

from pet_bootstrap import bootstrap

bootstrap(__file__)

from rich import print
from transformers import AutoTokenizer

from ProjectConfig import ProjectConfig

pc = ProjectConfig()


class Verbalizer(object):
    """
    Verbalizer クラス。1 つの Label とその子 Label の対応を管理する。
    """

    def __init__(self, verbalizer_file: str, tokenizer, max_label_len: int):
        """
        Args:
            verbalizer_file (str): verbalizer ファイルのパス。
            tokenizer: テキストと id の相互変換用トークナイザ。
            max_label_len (int): ラベル長。長い場合は切り詰め、短い場合はパディング。
        """
        self.tokenizer = tokenizer
        self.label_dict = self.load_label_dict(verbalizer_file)
        self.max_label_len = max_label_len

    def load_label_dict(self, verbalizer_file: str):
        """
        ローカルファイルを読み込み、verbalizer 辞書を構築する。
        Args:
            verbalizer_file (str): verbalizer ファイルのパス。
        Returns:
            dict -> {
                '体育': ['篮球', '足球','网球', '排球',  ...],
                '酒店': ['宾馆', '旅馆', '旅店', '酒店', ...],
                ...
            }
        """
        label_dict = {}
        with open(verbalizer_file, 'r', encoding='utf8') as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    label, sub_labels = line.split('\t', 1)
                except ValueError as exc:
                    raise ValueError(
                        f"verbalizer の {line_number} 行目は "
                        "「親ラベル<TAB>子ラベル,...」形式ではありません。"
                    ) from exc
                # 空要素を除き、記載順を保ったまま重複を除去する。
                label_dict[label] = list(
                    dict.fromkeys(value for value in sub_labels.split(',') if value)
                )
        return label_dict

    def find_sub_labels(self, label: Union[list, str]):
        """
        ラベルから対応するすべての子ラベルを取得する。

        Args:
            label (Union[list, str]): ラベル（テキストまたは id_list）, e.g. -> '体育' or [860, 5509]

        Returns:
            dict -> {
                'sub_labels': ['足球', '网球'],
                'token_ids': [[6639, 4413], [5381, 4413]]
            }
        """
        if type(label) == list:  # id_list の場合は tokenizer でテキストへ変換
            while self.tokenizer.pad_token_id in label:
                label.remove(self.tokenizer.pad_token_id)
            label = ''.join(self.tokenizer.convert_ids_to_tokens(label))
        if label not in self.label_dict:
            raise ValueError(f'Label Error: "{label}" not in label_dict')

        sub_labels = self.label_dict[label]
        ret = {'sub_labels': sub_labels}
        token_ids = [_id[1:-1] for _id in self.tokenizer(sub_labels)['input_ids']]
        for i in range(len(token_ids)):
            token_ids[i] = token_ids[i][:self.max_label_len]  # 切り詰めとパディング
            if len(token_ids[i]) < self.max_label_len:
                token_ids[i] = token_ids[i] + [self.tokenizer.pad_token_id] * (self.max_label_len - len(token_ids[i]))
        ret['token_ids'] = token_ids
        return ret

    def batch_find_sub_labels(self, label: List[Union[list, str]]):
        """
        子ラベルをバッチで取得する。

        Args:
            label (List[list, str]): ラベルリスト, [[4510, 5554], [860, 5509]] or ['体育', '电脑']

        Returns:
            list -> [
                        {
                         'sub_labels': ['足球', '网球'],
                                 'token_ids': [[6639, 4413], [5381, 4413]]
                        },
                        ...
                    ]
        """
        return [self.find_sub_labels(l) for l in label]

    def get_common_sub_str(self, str1: str, str2: str):
        """
        最長共通部分文字列を求める（1 次元 DP 最適化版）。
        時間計算量：O(m * n)
        空間計算量：O(min(m, n))
        """
        if not str1 or not str2:
            return "", 0

        # 常に short_str を短く保ち、メモリを抑える
        if len(str1) < len(str2):
            long_str, short_str = str2, str1
        else:
            long_str, short_str = str1, str2
        dp = [0] * (len(short_str) + 1)
        max_len = 0
        end_pos = 0
        for i in range(1, len(long_str) + 1):
            prev = 0
            for j in range(1, len(short_str) + 1):
                temp = dp[j]
                if long_str[i - 1] == short_str[j - 1]:
                    dp[j] = prev + 1
                    if dp[j] > max_len:
                        max_len = dp[j]
                        end_pos = i
                else:
                    dp[j] = 0
                prev = temp
        result = long_str[end_pos - max_len:end_pos]
        return result, max_len

    def hard_mapping(self, sub_label: str):
        """
        強マッチ関数。モデルが生成した子 label が存在しないとき、
        最長共通部分文字列で重なり最大の親 label を返す。

        Args:
            sub_label (str): 子 label。

        Returns:
            str: 親 label。
        """
        label, max_overlap = '无', 0
        for main_label, sub_labels in self.label_dict.items():
            overlap_num = 0
            for s_label in sub_labels:  # 全子 label と推論 label の最長共通部分文字列長を合算
                overlap_num += self.get_common_sub_str(sub_label, s_label)[1]
            if overlap_num > max_overlap:
                max_overlap = overlap_num
                label = main_label
        return label

    def find_main_label(self, sub_label: List[Union[list, str]], hard_mapping=True):
        """
        子ラベルから親ラベルを探す。

        Args:
            sub_label (List[Union[list, str]]): 子ラベル（テキストまたは id_list）, e.g. -> '苹果' or [5741, 3362]
            hard_mapping (bool): 生成語が存在しないとき、最も類似した label へ必ずマッピングするか。

        Returns:
            dict -> {
                'label': '水果',
                'token_ids': [3717, 3362]
            }
        """
        if type(sub_label) == list:  # id_list の場合は tokenizer でテキストへ戻す
            pad_token_id = self.tokenizer.pad_token_id
            while pad_token_id in sub_label:  # [PAD] token を除去
                sub_label.remove(pad_token_id)
            sub_label = ''.join(self.tokenizer.convert_ids_to_tokens(sub_label))
        main_label = '无'
        for label, s_labels in self.label_dict.items():
            if sub_label in s_labels:
                main_label = label
                break

        if main_label == '无' and hard_mapping:
            main_label = self.hard_mapping(sub_label)
        ret = {
            'label': main_label,
            'token_ids': self.tokenizer(main_label)['input_ids'][1:-1]
        }
        return ret

    def batch_find_main_label(self, sub_label: List[Union[list, str]], hard_mapping=True):
        """
        子ラベルから親ラベルをバッチで探す。

        Args:
            sub_label (List[Union[list, str]]): 子ラベルリスト, ['苹果', ...] or [[5741, 3362], ...]

        Returns:
            list: [
                    {
                    'label': '水果',
                    'token_ids': [3717, 3362]
                    },
                    ...
            ]
        """
        return [self.find_main_label(l, hard_mapping) for l in sub_label]


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained(pc.pre_model)
    verbalizer = Verbalizer(
        verbalizer_file=pc.verbalizer,
        tokenizer=tokenizer,
        max_label_len=pc.max_label_len
    )
    # demo 1: テキスト親ラベル -> 子ラベル + token_ids
    label = '衣服'
    ret = verbalizer.find_sub_labels(label)
    print(f'label={label}')
    print(ret)
    # 期待する出力の例:
    # {
    #   'sub_labels': ['衣服', '裙子', '西装', '袜子', '裤子', '袖子'],
    #   'token_ids': [[..., ...], ...]
    # }
    # demo 2: id リスト親ラベル -> 先にテキストへデコードしてから子ラベルを照会
    label_ids = tokenizer('水果', add_special_tokens=False)['input_ids']
    ret2 = verbalizer.find_sub_labels(label_ids)
    print(f'label_ids={label_ids}')
    print(ret2)

import json
# 返される文字列には例外の詳細情報が含まれる
import traceback
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
from functools import partial
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from glm_config import *


def convert_example(
        examples: dict,
        tokenizer,
        max_source_seq_len: int,
        max_target_seq_len: int,
):
    """
    サンプルデータをPtuningモデルが受け取る入力データに変換する。

    Args:
        examples (dict): 学習データのサンプル, e.g. -> {
                                                "text": [
                                                            '{"context": "年間基準金利は4.35%。実際には...", "target": "2017年の銀行貸出基準金利"}',
                                                            ...
                                                ]
                                            }
        max_source_seq_len (int): promptの最大長
        max_target_seq_len (int): 回答の最大長

    Returns:
        dict (str: np.array) -> tokenized_output = {
                            'input_ids': [[1525, 10, ...], [758, 2345, ...]],
                            'labels': [[822, 10, ...], [125, 58...]]
                        }
    """
    tokenized_output = {
        'input_ids': [],
        'labels': []
    }

    max_seq_length = max_source_seq_len + max_target_seq_len

    for example in examples['text']:
        try:
            example = json.loads(example)
            context = example["context"]
            target = example["target"]
            # print(f'context-->{context}')
            # print(f'target-->{target}')
            prompts_ids = tokenizer.encode(
                text=context,
                add_special_tokens=False
            )
            # print(f'prompts_ids--》{prompts_ids}\n{len(prompts_ids)}')

            target_ids = tokenizer.encode(
                text=target,
                add_special_tokens=False
            )
            # print(f'target_ids--》{target_ids}\n{len(target_ids)}')

            if len(prompts_ids) >= max_source_seq_len:
                # sourceの末尾に[gMASK] token用の位置を1つ確保する
                prompts_ids = prompts_ids[:max_source_seq_len - 1]

            if len(target_ids) >= max_target_seq_len - 1:
                # targetの先頭に<sop>、末尾に<eop> token用の位置を確保する
                target_ids = target_ids[:max_target_seq_len - 2]

                # source_ids + [gMASK] + <sop> + target_ids + <eop>
            input_ids = tokenizer.build_inputs_with_special_tokens(prompts_ids, target_ids)
            # print(f'input_ids-->{input_ids}')

            # bosはtargetの先頭にある
            context_length = input_ids.index(tokenizer.bos_token_id)
            # print(f'context_length-->{context_length}')
            # [gMASK]はsourceの末尾にある
            mask_position = context_length - 1

            # bosからeosまでのすべてのtargetをlabelとする
            labels = [-100] * context_length + input_ids[mask_position + 1:]
            # print(f'labels-->{labels}')

            pad_len = max_seq_length - len(input_ids)
            # print(f'pad_len-->{pad_len}')

            input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
            # print(f'input_ids-->{input_ids}\n{len(input_ids)}')
            labels = labels + [-100] * pad_len
            # print(f'labels-->{labels}\n{len(labels)}')

            tokenized_output['input_ids'].append(input_ids)
            tokenized_output['labels'].append(labels)
        except Exception:
            print(f'"{example}" -> {traceback.format_exc()}')
            continue

    for k, v in tokenized_output.items():
        tokenized_output[k] = np.array(v)

    return tokenized_output



def get_max_length(
        tokenizer,
        dataset_file: str
    ):
    """
    テストデータセットの入出力tokenの最大数を確認する。

    Args:
        dataset_file (str): _description_
    """
    source_seq_len_list = []
    target_seq_len_list = []
    with open(dataset_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            line = json.loads(line)

            source_len = tokenizer.encode(line['context'])
            source_seq_len_list.append(len(source_len))

            target_len = tokenizer.encode(line['target'])
            target_seq_len_list.append(len(target_len))

    print(dataset_file)
    print(f"【Source Sequence】 Max: {max(source_seq_len_list)}, Avg: {int(sum(source_seq_len_list) / len(source_seq_len_list))}, Middle: {sorted(source_seq_len_list)[int(len(source_seq_len_list) / 2)]}.")
    print(f"【Target Sequence】 Max: {max(target_seq_len_list)}, Avg: {int(sum(target_seq_len_list) / len(target_seq_len_list))}, Middle: {sorted(target_seq_len_list)[int(len(target_seq_len_list) / 2)]}.")


class _TestTokenizer:
    """ローカルの大規模モデルに依存せず、このファイルを自己テストするための最小tokenizer。"""

    bos_token_id = 2
    eos_token_id = 3
    pad_token_id = 0
    gmask_token_id = 1

    def encode(self, text, add_special_tokens=True):
        return [ord(char) + 10 for char in text]

    def build_inputs_with_special_tokens(self, prompt_ids, target_ids):
        return (
            prompt_ids
            + [self.gmask_token_id, self.bos_token_id]
            + target_ids
            + [self.eos_token_id]
        )


def main():
    """モデルファイルを使わずにconvert_exampleとget_max_lengthを簡易テストする。"""
    tokenizer = _TestTokenizer()
    examples = {
        'text': [
            json.dumps({'context': '输入', 'target': '输出'}, ensure_ascii=False),
            json.dumps({'context': '很长的测试输入', 'target': '很长的测试输出'}, ensure_ascii=False),
        ]
    }
    max_source_seq_len = 6
    max_target_seq_len = 5

    result = convert_example(
        examples,
        tokenizer=tokenizer,
        max_source_seq_len=max_source_seq_len,
        max_target_seq_len=max_target_seq_len,
    )

    expected_shape = (
        len(examples['text']),
        max_source_seq_len + max_target_seq_len,
    )
    assert result['input_ids'].shape == expected_shape, (
        f"input_ids shape 异常: {result['input_ids'].shape}"
    )
    assert result['labels'].shape == expected_shape, (
        f"labels shape 异常: {result['labels'].shape}"
    )
    assert np.all(
        result['labels'][result['input_ids'] == tokenizer.pad_token_id] == -100
    ), "padding 部分的 label 应为 -100"

    with TemporaryDirectory() as temp_dir:
        dataset_file = Path(temp_dir) / 'test_dataset.jsonl'
        dataset_file.write_text(
            '\n'.join(examples['text']) + '\n',
            encoding='utf-8',
        )
        get_max_length(tokenizer, str(dataset_file))

    print('自测通过：convert_example 和 get_max_length 运行正常。')


if __name__ == '__main__':
    main()

    
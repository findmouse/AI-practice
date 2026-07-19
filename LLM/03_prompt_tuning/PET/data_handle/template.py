# coding:utf-8
import sys
from pathlib import Path

for _root in Path(__file__).resolve().parents:
    if (_root / "pet_bootstrap.py").is_file():
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))
        break
else:
    raise RuntimeError("Cannot find PET project root (pet_bootstrap.py)")

from pet_bootstrap import bootstrap

bootstrap(__file__)

import numpy as np
from rich import print
from transformers import AutoTokenizer

from ProjectConfig import ProjectConfig


class HardTemplate(object):
    """
    ハードテンプレート。文と [MASK] の位置関係を人手で定義する。
    """

    def __init__(self, prompt: str):
        """
        Args:
            prompt (str): プロンプト形式の定義文字列, e.g. -> "这是一条{MASK}评论：{textA}。"
        """
        self.prompt = prompt
        self.inputs_list = []
        self.custom_tokens = set(['MASK'])
        self.prompt_analysis()

    def prompt_analysis(self):
        """
        プロンプト文字列テンプレートをマッピング可能なデータ構造へ分解する。

        Examples:
         prompt -> "这是一条{MASK}评论：{textA}。"
         inputs_list -> ['这', '是', '一', '条', 'MASK', '评', '论', '：', 'textA', '。']
         custom_tokens -> {'textA', 'MASK'}
        """
        if self.prompt.count('{') != self.prompt.count('}'):
            raise ValueError("Prompt の波括弧が対応していません。")

        idx = 0
        while idx < len(self.prompt):
            str_part = ''
            if self.prompt[idx] not in ['{', '}']:
                self.inputs_list.append(self.prompt[idx])
            if self.prompt[idx] == '{':
                idx += 1
                while idx < len(self.prompt) and self.prompt[idx] != '}':
                    str_part += self.prompt[idx]
                    idx += 1
                if idx == len(self.prompt):
                    raise ValueError("Prompt の '{' に対応する '}' がありません。")
            elif self.prompt[idx] == '}':
                raise ValueError("Unmatched bracket '}', check your prompt.")
            if str_part:
                self.inputs_list.append(str_part)
                self.custom_tokens.add(str_part)
            idx += 1

    def __call__(self,
                 inputs_dict: dict,
                 tokenizer,
                 mask_length,
                 max_seq_len=512):
        """
        1 サンプルを入力し、テンプレートに合う形式へ変換する。
        """
        outputs = {
            'text': '',
            'input_ids': [],
            'token_type_ids': [],
            'attention_mask': [],
            'mask_position': []
        }

        str_formated = ''
        for value in self.inputs_list:
            if value in self.custom_tokens:
                if value == 'MASK':
                    str_formated += inputs_dict[value] * mask_length
                else:
                    str_formated += inputs_dict[value]
            else:
                str_formated += value
        encoded = tokenizer(text=str_formated,
                            truncation=True,
                            max_length=max_seq_len,
                            padding='max_length')
        outputs['input_ids'] = encoded['input_ids']
        outputs['token_type_ids'] = encoded['token_type_ids']
        outputs['attention_mask'] = encoded['attention_mask']
        token_list = tokenizer.convert_ids_to_tokens(encoded['input_ids'])
        outputs['text'] = ''.join(token_list)
        mask_token_id = tokenizer.convert_tokens_to_ids(['[MASK]'])[0]
        condition = np.array(outputs['input_ids']) == mask_token_id
        mask_position = np.where(condition)[0].tolist()
        outputs['mask_position'] = mask_position
        return outputs


if __name__ == '__main__':
    pc = ProjectConfig()
    tokenizer = AutoTokenizer.from_pretrained(pc.pre_model)
    hard_template = HardTemplate(prompt='这是一条{MASK}评论：{textA}')
    print(hard_template.inputs_list)
    print(hard_template.custom_tokens)
    tep = hard_template(
                inputs_dict={'textA': '包装不错，苹果挺甜的，个头也大。', 'MASK': '[MASK]'},
                tokenizer=tokenizer,
                max_seq_len=30,
                mask_length=2)
    print(f"tep-->{tep}")

    print(tokenizer.convert_ids_to_tokens([3819, 3352]))
    print(tokenizer.convert_tokens_to_ids(['水', '果']))

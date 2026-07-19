# coding:utf-8
import sys
from functools import partial
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
import torch
from datasets import load_dataset
from rich import print
from transformers import AutoTokenizer

from ProjectConfig import ProjectConfig
from data_handle.template import HardTemplate


def convert_example(
        examples: dict,
        tokenizer,
        max_seq_len: int,
        max_label_len: int,
        hard_template: HardTemplate,
        train_mode=True,
        return_tensor=False) -> dict:
    """
    サンプルデータをモデル入力形式へ変換する。

    Args:
        examples (dict): 学習データサンプル, e.g. -> {
                                                "text": [
                                                            '手机\t这个手机也太卡了。',
                                                            '体育\t世界杯为何迟迟不见宣传',
                                                            ...
                                                ]
                                            }
        max_seq_len (int): 文の最大長。足りない場合は最大長まで padding
        max_label_len (int): label の最大長。足りない場合は最大長まで padding
        hard_template (HardTemplate): テンプレートクラス。
        train_mode (bool): 学習段階か推論段階か。
        return_tensor (bool): tensor を返すか。False なら numpy。

    Returns:
        dict (str: np.array) -> tokenized_output = {
                            'input_ids': [[1, 47, 10, 7, 304, 3, 3, 3, 3, 47, 27,
                                                        247, 98, 105, 512, 777, 15, 12043, 2], ...],
                            'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                                    0, 0, 0, 0, 0, 0, 0, 0], ...],
                            'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                                                    1, 1, 1, 1, 1, 1, 1, 1], ...],
                            'mask_positions': [[5, 6, 7, 8], ...],
                            'mask_labels': [[2372, 3442, 0, 0],
                                                            [2643, 4434, 2334, 0], ...]
                        }
    """
    tokenized_output = {
        'input_ids': [],
        'token_type_ids': [],
        'attention_mask': [],
        'mask_positions': [],
        'mask_labels': []
    }

    for example in examples['text']:
        if train_mode:
            try:
                label, content = example.strip().split('\t', 1)
            except ValueError as exc:
                raise ValueError(
                    "学習データは「ラベル<TAB>本文」形式である必要があります。"
                ) from exc
        else:
            content = example.strip()

        inputs_dict = {
            'textA': content,
            'MASK': '[MASK]'
        }
        encoded_inputs = hard_template(
            inputs_dict=inputs_dict,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            mask_length=max_label_len)
        tokenized_output['input_ids'].append(encoded_inputs["input_ids"])
        tokenized_output['token_type_ids'].append(encoded_inputs["token_type_ids"])
        tokenized_output['attention_mask'].append(encoded_inputs["attention_mask"])
        tokenized_output['mask_positions'].append(encoded_inputs["mask_position"])

        if train_mode:
            label_encoded = tokenizer(text=[label])
            label_encoded = label_encoded['input_ids'][0][1:-1]
            label_encoded = label_encoded[:max_label_len]
            add_pad = [tokenizer.pad_token_id] * (max_label_len - len(label_encoded))
            label_encoded = label_encoded + add_pad
            tokenized_output['mask_labels'].append(label_encoded)

    for k, v in tokenized_output.items():
        if return_tensor:
            tokenized_output[k] = torch.LongTensor(v)
        else:
            tokenized_output[k] = np.array(v)

    return tokenized_output


if __name__ == '__main__':
    pc = ProjectConfig()
    train_dataset = load_dataset('text', data_files=pc.train_path)
    print(type(train_dataset))
    print(train_dataset)
    tokenizer = AutoTokenizer.from_pretrained(pc.pre_model)
    hard_template = HardTemplate(prompt='这是一条{MASK}评论：{textA}')

    convert_func = partial(convert_example,
                           tokenizer=tokenizer,
                           hard_template=hard_template,
                           max_seq_len=30,
                           max_label_len=2)
    dataset = train_dataset.map(convert_func, batched=True)
    for value in dataset['train']:
        print(value)
        print(len(value['input_ids']))
        break

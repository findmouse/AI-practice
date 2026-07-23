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


def encode_text(tokenizer, text):
    """不经过 transformers 5.x 的 padding 流程进行纯文本编码。"""
    sentencepiece_tokenizer = getattr(tokenizer, "tokenizer", None)
    if sentencepiece_tokenizer is not None and hasattr(sentencepiece_tokenizer, "encode"):
        return sentencepiece_tokenizer.encode(text)
    return tokenizer.encode(text=text, add_special_tokens=False)


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
    skipped = 0
    first_error = None

    for example in examples['text']:
        try:
            example = json.loads(example)
            context = example["context"]
            target = example["target"]
            prompts_ids = encode_text(tokenizer, context)
            target_ids = encode_text(tokenizer, target)

            # ChatGLM2の特殊tokenは先頭の[gMASK]、sopと末尾のeos。
            prefix_length = len(tokenizer.get_prefix_tokens())
            prompts_ids = prompts_ids[:max_source_seq_len - prefix_length]
            target_ids = target_ids[:max_target_seq_len - 1]

            # [gMASK] + sop + source_ids + target_ids + eos
            input_ids = tokenizer.build_inputs_with_special_tokens(prompts_ids, target_ids)

            # ChatGLM2ではbos_token_idが定義されていないため、
            # prefixとsourceの長さからtargetの開始位置を求める。
            context_length = prefix_length + len(prompts_ids)

            # prefixとsourceをmaskし、targetからeosまでを学習対象にする。
            labels = [-100] * context_length + input_ids[context_length:]

            pad_len = max_seq_length - len(input_ids)
            input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
            labels = labels + [-100] * pad_len

            tokenized_output['input_ids'].append(np.array(input_ids))
            tokenized_output['labels'].append(np.array(labels))
        except TypeError as exc:
            # transformers 5.x 与旧 ChatGLMTokenizer._pad 不兼容时直接失败，避免刷屏。
            if "padding_side" in str(exc):
                raise RuntimeError(
                    "ChatGLMTokenizer 与当前 transformers 不兼容（_pad 缺少 padding_side）。"
                    "请确认 train.py 已调用 patch_chatglm_tokenizer。"
                ) from exc
            skipped += 1
            if first_error is None:
                first_error = traceback.format_exc()
            continue
        except Exception:
            skipped += 1
            if first_error is None:
                first_error = traceback.format_exc()
            continue

    if not tokenized_output['input_ids']:
        detail = first_error or "未知错误"
        raise RuntimeError(
            f"当前批次没有可用样本（跳过 {skipped} 条）。首个错误:\n{detail}"
        )
    if skipped:
        print(f"警告: 本批次跳过 {skipped} 条样本。首个错误:\n{first_error}")

    for k, v in tokenized_output.items():
        tokenized_output[k] = np.stack(v)

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

            source_len = encode_text(tokenizer, line['context'])
            source_seq_len_list.append(len(source_len))

            target_len = encode_text(tokenizer, line['target'])
            target_seq_len_list.append(len(target_len))

    print(dataset_file)
    print(f"【Source Sequence】 Max: {max(source_seq_len_list)}, Avg: {int(sum(source_seq_len_list) / len(source_seq_len_list))}, Middle: {sorted(source_seq_len_list)[int(len(source_seq_len_list) / 2)]}.")
    print(f"【Target Sequence】 Max: {max(target_seq_len_list)}, Avg: {int(sum(target_seq_len_list) / len(target_seq_len_list))}, Middle: {sorted(target_seq_len_list)[int(len(target_seq_len_list) / 2)]}.")


class _TestTokenizer:
    """ChatGLM2の特殊token配置を再現する自己テスト用tokenizer。"""

    sop_token_id = 2
    eos_token_id = 3
    pad_token_id = 0
    gmask_token_id = 1

    def encode(self, text, add_special_tokens=True):
        return [ord(char) + 10 for char in text]

    def get_prefix_tokens(self):
        return [self.gmask_token_id, self.sop_token_id]

    def build_inputs_with_special_tokens(self, prompt_ids, target_ids):
        return (
            self.get_prefix_tokens()
            + prompt_ids
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
    max_source_seq_len = 60
    max_target_seq_len = 50

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

    
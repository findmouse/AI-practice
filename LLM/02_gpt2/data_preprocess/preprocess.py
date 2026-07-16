# -*- coding: utf-8 -*-
"""
medical_*.txt の対話コーパスを tokenize し、pkl として保存する。
各対話の形式: [CLS] utt1 [SEP] utt2 [SEP] ...
"""
import os
import pickle
import sys

from tqdm import tqdm
from transformers import BertTokenizerFast

# プロジェクトルートまたは本ディレクトリから直接実行できるようにする
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from parameter_config import ParameterConfig


def preprocess(txt_path, pkl_path, vocab_path):
    """
    元コーパスをトークン化し、各対話を次の形式に変換する:
    [CLS] utterance1 [SEP] utterance2 [SEP] ...
    """
    tokenizer = BertTokenizerFast(
        vocab_file=vocab_path,
        sep_token='[SEP]',
        pad_token='[PAD]',
        cls_token='[CLS]',
    )
    sep_id = tokenizer.sep_token_id
    cls_id = tokenizer.cls_token_id

    with open(txt_path, 'rb') as f:
        data = f.read().decode('utf-8')

    if '\r\n' in data:
        train_data = data.split('\r\n\r\n')
    else:
        train_data = data.split('\n\n')

    dialogue_list = []
    dialogue_len = []
    for dialogue in tqdm(train_data, desc=os.path.basename(txt_path)):
        dialogue = dialogue.strip()
        if not dialogue:
            continue
        if '\r\n' in dialogue:
            sequences = dialogue.split('\r\n')
        else:
            sequences = dialogue.split('\n')

        input_ids = [cls_id]
        for sequence in sequences:
            sequence = sequence.strip()
            if not sequence:
                continue
            input_ids += tokenizer.encode(sequence, add_special_tokens=False)
            input_ids.append(sep_id)

        if len(input_ids) > 1:
            dialogue_len.append(len(input_ids))
            dialogue_list.append(input_ids)

    os.makedirs(os.path.dirname(pkl_path) or '.', exist_ok=True)
    with open(pkl_path, 'wb') as f:
        pickle.dump(dialogue_list, f)

    avg_len = sum(dialogue_len) / len(dialogue_len) if dialogue_len else 0
    print(f'保存完了: {pkl_path}')
    print(f'対話数: {len(dialogue_list)}, 平均長: {avg_len:.1f}')
    return dialogue_list


def main():
    params = ParameterConfig()
    print(f'語彙パス: {params.vocab_path}')
    print('学習データの処理を開始します...')
    preprocess(params.train_txt_path, params.train_path, params.vocab_path)
    print('検証データの処理を開始します...')
    preprocess(params.valid_txt_path, params.valid_path, params.vocab_path)
    print('データ前処理がすべて完了しました。')


if __name__ == '__main__':
    main()

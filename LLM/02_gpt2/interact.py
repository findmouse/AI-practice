# -*- coding: utf-8 -*-
"""
医療問診ボットの対話テスト。
実行方法（02_gpt2 ディレクトリで）:
    python interact.py
quit / exit / Ctrl+C で終了。会話履歴は sample/samples.txt に保存される。
"""
import os
from datetime import datetime

import torch
import torch.nn.functional as F
from transformers import BertTokenizerFast, GPT2LMHeadModel

from parameter_config import ParameterConfig


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """top-k / top-p で logits をフィルタする。"""
    assert logits.dim() == 1
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def resolve_model_path(params):
    candidates = [
        os.path.join(params.save_model_path, 'min_ppl_model'),
        os.path.join(params.save_model_path, 'epoch{}'.format(params.epochs)),
        os.path.join(params.project_root, 'save_model1', 'min_ppl_model_bj'),
    ]
    # 教材コードの旧命名にも対応
    if os.path.isdir(params.save_model_path):
        for name in sorted(os.listdir(params.save_model_path), reverse=True):
            path = os.path.join(params.save_model_path, name)
            if os.path.isdir(path):
                candidates.append(path)

    for path in candidates:
        config_file = os.path.join(path, 'config.json')
        if os.path.exists(config_file):
            return path
    raise FileNotFoundError(
        '学習済みモデルが見つかりません。先に train.py を実行してください。\n'
        f'想定パス例: {os.path.join(params.save_model_path, "min_ppl_model")}'
    )


def generate_reply(model, tokenizer, history, params, device):
    input_ids = [tokenizer.cls_token_id]
    for history_utr in history[-params.max_history_len:]:
        input_ids.extend(history_utr)
        input_ids.append(tokenizer.sep_token_id)

    input_ids = torch.tensor(input_ids).long().to(device).unsqueeze(0)
    response = []
    max_gen_len = min(params.max_len, 100)

    for _ in range(max_gen_len):
        outputs = model(input_ids=input_ids)
        next_token_logits = outputs.logits[0, -1, :].clone()

        for token_id in set(response):
            next_token_logits[token_id] /= params.repetition_penalty
        unk_id = tokenizer.convert_tokens_to_ids('[UNK]')
        next_token_logits[unk_id] = -float('Inf')

        filtered_logits = top_k_top_p_filtering(
            next_token_logits, top_k=params.topk, top_p=params.topp
        )
        next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
        if next_token.item() == tokenizer.sep_token_id:
            break
        response.append(next_token.item())
        input_ids = torch.cat((input_ids, next_token.unsqueeze(0)), dim=1)

    return response


def main():
    params = ParameterConfig()
    device = params.device
    print('using device: {}'.format(device))

    tokenizer = BertTokenizerFast(
        vocab_file=params.vocab_path,
        sep_token='[SEP]',
        pad_token='[PAD]',
        cls_token='[CLS]',
    )

    model_path = resolve_model_path(params)
    print(f'モデルを読み込みます: {model_path}')
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model = model.to(device)
    model.eval()

    samples_file = None
    if params.save_samples_path:
        os.makedirs(params.save_samples_path, exist_ok=True)
        samples_file = open(
            os.path.join(params.save_samples_path, 'samples.txt'),
            'a',
            encoding='utf-8',
        )
        samples_file.write('チャット履歴 {}:\n'.format(datetime.now()))

    history = []
    print('医療問診アシスタントとのチャットを開始します（quit/exit または Ctrl+C で終了）')

    try:
        while True:
            text = input('user:').strip()
            if not text:
                continue
            if text.lower() in {'quit', 'exit', 'q'}:
                break

            if samples_file:
                samples_file.write('user:{}\n'.format(text))

            text_ids = tokenizer.encode(text, add_special_tokens=False)
            history.append(text_ids)
            response = generate_reply(model, tokenizer, history, params, device)
            history.append(response)
            reply = ''.join(tokenizer.convert_ids_to_tokens(response))
            print('chatbot:' + reply)

            if samples_file:
                samples_file.write('chatbot:{}\n'.format(reply))
    except KeyboardInterrupt:
        print('\n対話を終了します')
    finally:
        if samples_file:
            samples_file.close()


if __name__ == '__main__':
    main()

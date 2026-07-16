# -*- coding: utf-8 -*-
"""
デモ一括実行フロー:
1) データ前処理
2) 高速学習（デフォルトは少量サンプルでフロー確認）
3) 損失曲線の描画
4) 任意で対話テストへ移行

使い方（02_gpt2 ディレクトリで）:
    python run_pipeline.py
    python run_pipeline.py --full
    python run_pipeline.py --interact
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data_preprocess.dataloader import get_dataloader
from data_preprocess.preprocess import preprocess
from parameter_config import ParameterConfig
from train import ensure_pkl_data, train
from transformers import BertTokenizerFast, GPT2Config, GPT2LMHeadModel


def parse_args():
    parser = argparse.ArgumentParser(description='GPT2 医療問診ボット demo フロー')
    parser.add_argument('--full', action='store_true', help='全学習データを使用する')
    parser.add_argument('--epochs', type=int, default=None, help='学習エポック数')
    parser.add_argument('--batch-size', type=int, default=None, help='batch size')
    parser.add_argument('--max-train-samples', type=int, default=200,
                        help='高速モード時の学習サンプル数上限')
    parser.add_argument('--max-valid-samples', type=int, default=50,
                        help='高速モード時の検証サンプル数上限')
    parser.add_argument('--interact', action='store_true', help='学習後に対話テストへ進む')
    return parser.parse_args()


def build_params(args):
    params = ParameterConfig()
    if args.full:
        params.max_train_samples = None
        params.max_valid_samples = None
        params.epochs = args.epochs or params.epochs
        params.batch_size = args.batch_size or params.batch_size
        print('全データ学習モード')
    else:
        params.max_train_samples = args.max_train_samples
        params.max_valid_samples = args.max_valid_samples
        params.epochs = args.epochs or 1
        params.batch_size = args.batch_size or 2
        params.loss_step = 20
        print(
            f'高速モード: train_samples={params.max_train_samples}, '
            f'valid_samples={params.max_valid_samples}, '
            f'epochs={params.epochs}, batch_size={params.batch_size}'
        )
    return params


def run_train(params):
    print(f'使用デバイス: {params.device}')
    os.makedirs(params.save_model_path, exist_ok=True)
    ensure_pkl_data(params)

    tokenizer = BertTokenizerFast(
        vocab_file=params.vocab_path,
        sep_token='[SEP]',
        pad_token='[PAD]',
        cls_token='[CLS]',
    )

    if params.pretrained_model:
        model = GPT2LMHeadModel.from_pretrained(params.pretrained_model)
    else:
        model_config = GPT2Config.from_json_file(params.config_json)
        model = GPT2LMHeadModel(config=model_config)

    model = model.to(params.device)
    assert model.config.vocab_size == tokenizer.vocab_size
    print(f'モデルのパラメータ総数: {sum(p.numel() for p in model.parameters())}')

    train_dataloader, validate_dataloader = get_dataloader(
        params.train_path,
        params.valid_path,
        batch_size=params.batch_size,
        max_len=params.max_len,
        max_train_samples=params.max_train_samples,
        max_valid_samples=params.max_valid_samples,
    )
    train(model, train_dataloader, validate_dataloader, params)


def main():
    args = parse_args()
    params = build_params(args)

    print('======== 1. データ前処理 ========')
    if not os.path.exists(params.train_path):
        preprocess(params.train_txt_path, params.train_path, params.vocab_path)
    else:
        print(f'既に存在します: {params.train_path}')
    if not os.path.exists(params.valid_path):
        preprocess(params.valid_txt_path, params.valid_path, params.vocab_path)
    else:
        print(f'既に存在します: {params.valid_path}')

    print('======== 2. モデル学習 ========')
    run_train(params)

    print('======== 3. 損失曲線 ========')
    print(f'確認してください: {params.loss_curve_path}')
    print(f'損失の数値: {params.loss_history_path}')

    if args.interact:
        print('======== 4. 機能テスト（対話） ========')
        from interact import main as interact_main
        interact_main()
    else:
        print('学習が完了しました。機能テストは: python interact.py')
        print('損失曲線の再確認は: python plot_loss.py')


if __name__ == '__main__':
    main()

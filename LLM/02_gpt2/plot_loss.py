# -*- coding: utf-8 -*-
"""loss_history.json を読み込み、学習/検証損失曲線を描画する。"""
import json
import os
import sys

import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from parameter_config import ParameterConfig


def plot_loss(history_path=None, save_path=None, show=True):
    params = ParameterConfig()
    history_path = history_path or params.loss_history_path
    save_path = save_path or params.loss_curve_path

    if not os.path.exists(history_path):
        raise FileNotFoundError(
            f'損失履歴ファイルが見つかりません: {history_path}\n先に train.py で学習を完了してください。'
        )

    with open(history_path, 'r', encoding='utf-8') as f:
        history = json.load(f)

    train_losses = history.get('train_losses', [])
    validate_losses = history.get('validate_losses', [])
    if not train_losses:
        raise ValueError('損失履歴が空のため、描画できません。')

    epochs = list(range(1, len(train_losses) + 1))
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, marker='o', label='train loss')
    if validate_losses:
        plt.plot(epochs, validate_losses, marker='s', label='valid loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('GPT2 Medical Chatbot Loss Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f'損失曲線を保存しました: {save_path}')
    if show:
        plt.show()
    else:
        plt.close()


if __name__ == '__main__':
    # GUI がなくても画像として保存可能
    plot_loss(show=False)

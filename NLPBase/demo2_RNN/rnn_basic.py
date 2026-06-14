# -*- coding: utf-8 -*-
"""
RNN（Recurrent Neural Network）の基本動作確認サンプル

学習内容
- RNNの基本構造
- sequence_lengthの影響
- num_layersの影響
- 出力テンソルの形状確認

Author: Hao Ding
"""

import torch
import torch.nn as nn


def rnn_basic_example() -> None:
    """
    RNNの基本動作確認

    Parameters
    ----------
    input_size : int
        入力ベクトル（単語埋め込み）の次元数

    hidden_size : int
        隠れ状態の次元数

    num_layers : int
        RNNレイヤ数
    """

    rnn = nn.RNN(
        input_size=5,
        hidden_size=6,
        num_layers=1
    )

    # 入力データ
    #
    # shape:
    # (sequence_length, batch_size, input_size)
    input_tensor = torch.randn(1, 3, 5)

    # 初期隠れ状態
    #
    # shape:
    # (num_layers, batch_size, hidden_size)
    h0 = torch.randn(1, 3, 6)

    output, hn = rnn(input_tensor, h0)

    print("=== Basic Example ===")
    print(f"Output Shape : {output.shape}")
    print(f"Hidden Shape : {hn.shape}")


def rnn_sequence_length_example() -> None:
    """
    sequence_lengthを変更した場合の確認
    """

    rnn = nn.RNN(
        input_size=5,
        hidden_size=6,
        num_layers=1
    )

    # sequence_length = 8
    input_tensor = torch.randn(8, 3, 5)

    h0 = torch.randn(1, 3, 6)

    output, hn = rnn(input_tensor, h0)

    print("=== Sequence Length Example ===")
    print(f"Output Shape : {output.shape}")
    print(f"Hidden Shape : {hn.shape}")


def rnn_multi_layer_example() -> None:
    """
    複数レイヤのRNN動作確認
    """
    # num_layers = 2
    rnn = nn.RNN(
        input_size=5,
        hidden_size=6,
        num_layers=2
    )

    input_tensor = torch.randn(4, 3, 5)

    h0 = torch.randn(2, 3, 6)

    output, hn = rnn(input_tensor, h0)

    print("=== Multi Layer Example ===")
    print(f"Output Shape : {output.shape}")
    print(f"Hidden Shape : {hn.shape}")


if __name__ == "__main__":
    # rnn_basic_example()
    # rnn_sequence_length_example()
    rnn_multi_layer_example()

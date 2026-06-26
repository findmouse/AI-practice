# -*- coding: utf-8 -*-
"""
システム名：自然言語処理・基盤AIシステム
モジュール名：transformer_input
作成日：2026/06/24
作成者：AI開発チーム
修正履歴：

"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class Embeddings(nn.Module):
    """単語埋め込み（Word Embedding）レイヤークラス。

    離散的な単語IDを、指定された次元数の密な分散表現（ベクトル）に変換します。
    """

    def __init__(self, vocab_size: int, d_model: int):
        """初期化処理。

        Args:
            vocab_size (int): 総語彙数（重複を除く単語の総数）
            d_model (int): 単語埋め込みベクトルの次元数
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        # 埋め込み層の定義
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """順伝播処理。

        Args:
            x (torch.Tensor): 入力トークンIDテンソル

        Returns:
            torch.Tensor: スケーリングされた埋め込みベクトル
        """
        if self.training:
            print(f"[DEBUG] Embeddings - 出力形状: {self.embed(x).shape}")

        # 後の位置エンコーディングとのバランスを保つため、次元数の平方根を乗算します
        return self.embed(x) * math.sqrt(self.d_model)


class PositionEncoding(nn.Module):
    """位置エンコーディング（Position Encoding）レイヤークラス。

    シーケンス内の単語の相対的・絶対的な位置情報を正弦波・余弦波を用いて付与します。
    """

    def __init__(self, d_model: int, dropout: float, max_len: int = 60):
        """初期化処理。

        Args:
            d_model (int): 隠れ層の次元数
            dropout (float): ドロップアウト率
            max_len (int): 許容する最大シーケンス長
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 位置エンコーディング行列の初期化 [max_len, d_model]
        pe = torch.zeros(max_len, d_model)

        # 位置インデックス列の生成 [max_len, 1]
        position = torch.arange(0, max_len).unsqueeze(dim=1)

        # 周波数変換用行列の計算 (公式: 1 / 10000^(2i/d_model))
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000) / d_model))

        # 各位置における角度の計算 [max_len, d_model/2]
        position_value = position * div_term

        # 偶数列にSin波、奇数列にCos波を適用
        pe[:, 0::2] = torch.sin(position_value)
        pe[:, 1::2] = torch.cos(position_value)

        # ミニバッチ処理に対応するため次元を拡張 [1, max_len, d_model]
        pe = pe.unsqueeze(0)

        # パラメータ更新の対象外（バックプロパゲーションをしないバッファ）として登録
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """順伝播处理。

        Args:
            x (torch.Tensor): 埋め込み層からの出力ベクトル [バッチサイズ, シーケンス長, 次元数]
        """
        # 入力シーケンス長に合わせて位置エンコーディングを切り出し、加算（ブロードキャスト）します
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


def test_position():
    """入力レイヤー群の単体テスト関数。"""
    vocab_size = 1000
    d_model = 512
    my_embed = Embeddings(vocab_size=vocab_size, d_model=d_model)
    x = torch.tensor([[100, 2, 412, 508],
                      [491, 998, 1, 221]])
    embed_x = my_embed(x)
    print(f"[INFO] Embeddings 出力形状: {embed_x.shape}")

    my_position = PositionEncoding(d_model, 0.1, max_len=60)
    position_x = my_position(embed_x)
    print(f"[INFO] PositionEncoding 出力形状: {position_x.shape}")


def dm_draw_PE_feature():
    """位置エンコーディングのSin-Cos特性曲線をプロット・保存する関数。"""
    my_pe = PositionEncoding(d_model=20, dropout=0, max_len=100)
    y = my_pe(torch.zeros((1, 100, 20)))

    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(100), y[0, :, 4:8].detach().numpy())
    plt.legend([f"dim {p}" for p in [4, 5, 6, 7]])
    plt.title("Position Encoding Sin-Cos Curves")

    img_dir = os.path.join(os.path.dirname(__file__), "img")
    os.makedirs(img_dir, exist_ok=True)
    save_path = os.path.join(img_dir, "pe_feature.png")
    plt.savefig(save_path, dpi=150)
    print(f"[SUCCESS] 画像を保存しました: {save_path}")
    plt.show()


if __name__ == '__main__':
    test_position()
    dm_draw_PE_feature()
# coding:utf-8
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomAttention(nn.Module):
    """カスタム注意（Attention）メカニズムを計算するクラス。

    クエリ(Q)とキー(K)を結合し、注意の重みを計算した後、
    バリュー(V)と行列を積行い、最終的な線形変換出力を生成します。
    """

    def __init__(
            self,
            query_size: int,
            key_size: int,
            value_size1: int,
            value_size2: int,
            output_size: int
    ) -> None:
        """初期化メソッド。

        Args:
            query_size (int): クエリ(Q)の機能次元数。
            key_size (int): キー(K)の機能次元数。
            value_size1 (int): バリュー(V)のシーケンス長（重みとの行列積用）。
            value_size2 (int): バリュー(V)の機能次元数。
            output_size (int): 最終出力の機能次元数。
        """
        super().__init__()
        self.query_size = query_size
        self.key_size = key_size
        self.value_size1 = value_size1
        self.value_size2 = value_size2
        self.output_size = output_size

        # スコア計算用全結合層: QとKを結合(concat)して入力するため、入力次元は query_size + key_size となる。
        # 出力次元を value_size1 にすることで、後続のValue行列との行列積(bmm)を可能にする。
        self.attn_linear = nn.Linear(self.query_size + self.key_size, self.value_size1)

        # 最終出力用全結合層: Qとコンテキストベクトル(temp)を結合して入力する。
        # 入力次元は query_size + value_size2、出力次元は指定された output_size となる。
        self.output_linear = nn.Linear(self.query_size + self.value_size2, self.output_size)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """順伝播（フォワードパス）の処理。

        Args:
            Q (torch.Tensor): クエリテンソル。形状: [Batch_Size, Seq_Len_Q, Query_Size]
            K (torch.Tensor): キーテンソル。形状: [Batch_Size, Seq_Len_K, Key_Size]
            V (torch.Tensor): バリューテンソル。形状: [Batch_Size, Value_Size1, Value_Size2]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - result: 注意機構の最終出力。形状: [Batch_Size, Seq_Len_Q, Output_Size]
                - attn_weight: 計算された注意の重み（スコア）。形状: [Batch_Size, Seq_Len_Q, Value_Size1]
        """
        # --- ステップ 1: QとKを結合し、全結合層とSoftmaxを適用して注意の重み(分数)を計算 ---
        # 形状変化: [B, S_Q, Q_Size] + [B, S_K, K_Size] -> [B, S_Q, Q_Size + K_Size]
        combined_qk = torch.cat((Q, K), dim=-1)

        # 形状変化: [B, S_Q, Q_Size + K_Size] -> [B, S_Q, Value_Size1]
        attn_weight = F.softmax(self.attn_linear(combined_qk), dim=-1)

        # --- ステップ 2: 注意の重みとVの行列積を計算（コンテキストベクトルの抽出） ---
        # 形状変化: [B, S_Q, Value_Size1] * [B, Value_Size1, Value_Size2] -> [B, S_Q, Value_Size2]
        context_vector = torch.bmm(attn_weight, V)

        # --- ステップ 3: 元のQとコンテキストベクトルを結合 ---
        # 形状変化: [B, S_Q, Query_Size] + [B, S_Q, Value_Size2] -> [B, S_Q, Query_Size + Value_Size2]
        combined_output = torch.cat((Q, context_vector), dim=-1)

        # --- ステップ 4: 最終的な線形変換を行い、指定された出力サイズに変形 ---
        # 形状変化: [B, S_Q, Query_Size + Value_Size2] -> [B, S_Q, Output_Size]
        result = self.output_linear(combined_output)

        return result, attn_weight


if __name__ == '__main__':
    # テンソルの定義 (Batch_Size=1, Seq_Len=1)
    query_tensor = torch.randn(1, 1, 32)  # 32 -> query_size
    key_tensor = torch.randn(1, 1, 32)  # 32 -> key_size
    value_tensor = torch.randn(1, 32, 64)  # 32 -> value_size1, 64 -> value_size2

    # インスタンス化
    attention_module = CustomAttention(
        query_size=32,
        key_size=32,
        value_size1=32,
        value_size2=64,
        output_size=32
    )

    # 推論の実行
    output_result, attention_weights = attention_module(query_tensor, key_tensor, value_tensor)

    # ログ出力
    print(f"output_result.shape     --> {output_result.shape}")
    print(f"attention_weights.shape --> {attention_weights.shape}")

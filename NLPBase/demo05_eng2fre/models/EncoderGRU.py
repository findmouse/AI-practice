import os
import sys
from typing import Tuple

# ネットワーク構造および関数構築のためのPyTorchライブラリ
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 自作モジュール用のインポートパス追加設定
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 実行デバイスの設定（GPUが利用可能であればCUDA、不可であればCPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 特殊トークンおよび制約条件の定義
SOS_TOKEN = 0  # Start Of Sequence (開始フラグ)
EOS_TOKEN = 1  # End Of Sequence (終了フラグ)
MAX_LENGTH = 10  # 扱う文章の最大単語数（記号含む）

# データファイルの参照パス
DATA_PATH = "../data/eng-fra-v2.txt"


class EncoderGRU(nn.Module):
    """GRUを採用したSeq2Seq用のエンコーダー（符号化器）モデルクラスです。"""

    def __init__(self, vocab_size: int, hidden_size: int):
        """初期化処理

        Args:
            vocab_size (int): ボキャブラリサイズ（重複を除いた単語の総数）
            hidden_size (int): 隠れ層および分散表現（Embedding）の次元数
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        # Embedding（単語埋め込み）層の定義
        self.embed = nn.Embedding(self.vocab_size, self.hidden_size)

        # GRU層の定義（batch_first=True により入出力を [バッチサイズ, シーケンス長, 次元数] に統一）
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)

    def forward(self, input_tensor: torch.Tensor, hidden_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """順伝播処理（フォワードパス）を実行します。

        Args:
            input_tensor (torch.Tensor): 入力シーケンスのIDテンソル [バッチサイズ, シーケンス長]
            hidden_tensor (torch.Tensor): 初期状態または前ステップの隠れ状態テンソル [レイヤー数, バッチサイズ, 隠れ層次元数]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - output (torch.Tensor): 各タイムステップの出力テンソル
                - hn (torch.Tensor): 最終タイムステップの隠れ状態テンソル
        """
        # [バッチサイズ, シーケンス長] -> [バッチサイズ, シーケンス長, 隠れ層次元数] へ埋め込み
        embedded = self.embed(input_tensor)

        # GRU層へデータを入力
        output, hn = self.gru(embedded, hidden_tensor)
        return output, hn

    def init_hidden(self, batch_size: int = 1) -> torch.Tensor:
        """ゼロ初期化したGRU隠れ状態を返します。"""
        return torch.zeros(
            1, batch_size, self.hidden_size, device=next(self.parameters()).device
        )
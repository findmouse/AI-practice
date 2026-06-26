import os
import sys
from typing import Tuple

# ネットワーク構造および関数構築のためのPyTorchライブラリ
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 自作モジュール用のインポートパス追加設定
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.EncoderGRU import EncoderGRU
from utils.dataset import TranslationDataset, get_data

# 実行デバイスの設定（GPUが利用可能であればCUDA、不可であればCPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 特殊トークンおよび制約条件の定義
SOS_TOKEN = 0  # Start Of Sequence (開始フラグ)
EOS_TOKEN = 1  # End Of Sequence (終了フラグ)
MAX_LENGTH = 10  # 扱う文章の最大単語数（記号含む）

# データファイルの参照パス
DATA_PATH = "../data/eng-fra-v2.txt"


class DecoderGRU(nn.Module):
    """アテンション非搭載の標準的なSeq2Seq用デコーダー（復号化器）モデルクラスです。"""

    def __init__(self, vocab_size: int, hidden_size: int):
        """初期化処理

        Args:
            vocab_size (int): 出力言語（フランス語）のボキャブラリサイズ（重複を除く単語総数）
            hidden_size (int): 隠れ層および分散表現（Embedding）の次元数
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        # Embedding（単語埋め込み）層の定義
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)

        # GRU層の定義（batch_first=True により入出力を [バッチサイズ, シーケンス長, 次元数] に統一）
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)

        # 出力層（線形変換層）：GRUの出力から各単語のスコア（ロジット）を算出
        self.linear = nn.Linear(self.hidden_size, self.vocab_size)

        # 確率分布へ変換するためのLogSoftmax層（損失関数に nn.NLLLoss を使用することを想定）
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input_tensor: torch.Tensor, hidden_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """順伝播処理（フォワードパス）を実行します。

        デコーダーは推論時、1タイムステップ（1単語）ずつ入力を受け取って予測を行います。

        Args:
            input_tensor (torch.Tensor): 現ステップの入力単語IDテンソル [バッチサイズ, 1]
            hidden_tensor (torch.Tensor): 前のステップ（またはエンコーダーの最終状態）の隠れ状態テンソル

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - output_probs (torch.Tensor): 各単語の対数確率分布
                - hidden (torch.Tensor): 次ステップへ引き継ぐ隠れ状態テンソル
        """
        # 単語IDを分散表現に変換: [バッチサイズ, 1] -> [バッチサイズ, 1, 隠れ層次元数]
        embedded = self.embedding(input_tensor)

        # GRU層へ入力データを転送
        output, hidden = self.gru(embedded, hidden_tensor)

        # 出力層およびLogSoftmaxの適用
        # output[0] によりシーケンス次元（通常1固定）のサイズを調整して線形層に適用
        results = self.linear(output[:, 0, :])
        output_probs = self.softmax(results)

        return output_probs, hidden


def main() -> None:
    """エンコーダーおよびデコーダーモデルの動作検証用メイン処理です。"""
    (
        _,
        _,
        eng_vocab_size,
        fra_vocab_size,
        _,
        _,
        pairs,
    ) = get_data()

    dataset = TranslationDataset(pairs)
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
    hidden_size = 256

    encoder_gru = EncoderGRU(eng_vocab_size, hidden_size).to(device)
    decoder_gru = DecoderGRU(fra_vocab_size, hidden_size).to(device)
    print(f"エンコーダーのモデルアーキテクチャ:\n{encoder_gru}\n")
    print(f"デコーダーのモデルアーキテクチャ:\n{decoder_gru}\n")

    for batch_x, _ in dataloader:
        batch_x = batch_x.to(device)
        encoder_hidden = encoder_gru.init_hidden(batch_size=batch_x.size(0))
        _, encoder_hidden = encoder_gru(batch_x, encoder_hidden)

        decoder_input = torch.full((batch_x.size(0), 1), SOS_TOKEN, dtype=torch.long, device=device)
        output_probs, decoder_hidden = decoder_gru(decoder_input, encoder_hidden)
        print(f"decoder output shape -> {output_probs.shape}")
        print(f"decoder hidden shape -> {decoder_hidden.shape}")
        break


if __name__ == "__main__":
    main()

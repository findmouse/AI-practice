import os
import sys
from typing import Tuple

# ネットワーク構造および関数構築のためのPyTorchライブラリ
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# 自作モジュール用のインポートパス追加設定
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.EncoderGRU import EncoderGRU
from utils.dataset import TranslationDataset, get_data

# 特殊トークンおよび制約条件の定義
MAX_LENGTH = 10  # 扱う文章の最大単語数（記号含む）
SOS_TOKEN = 0  # Start Of Sequence (開始フラグ)
EOS_TOKEN = 1  # End Of Sequence (終了フラグ)

# 実行デバイスの設定（GPUが利用可能であればCUDA、不可であればCPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_encoder_outputs(encoder_output: torch.Tensor, max_len: int = MAX_LENGTH) -> torch.Tensor:
    """エンコーダーの出力シーケンスを固定長にパディングします。

    アテンション付きデコーダーの Key / Value ベクトルとして利用するため、形状を揃えます。

    Args:
        encoder_output (torch.Tensor): エンコーダーからの出力テンソル [バッチサイズ, 入力シーケンス長, 隠れ層次元数]
        max_len (int, optional): パディング後の最大シーケンス長. Defaults to MAX_LENGTH.

    Returns:
        torch.Tensor: パディング済みのエンコーダー出力テンソル [バッチサイズ, 最大シーケンス長, 隠れ層次元数]
    """
    batch_size = encoder_output.size(0)
    hidden_size = encoder_output.size(-1)
    seq_len = min(encoder_output.size(1), max_len)

    # ゼロで初期化した固定長テンソルを作成し、実データをコピー
    encoder_outputs = torch.zeros(
        batch_size, max_len, hidden_size, device=encoder_output.device
    )
    encoder_outputs[:, :seq_len] = encoder_output[:, :seq_len]
    return encoder_outputs


class AttentionDecoder(nn.Module):
    """アテンション（注意機構）を搭載したSeq2Seq用デコーダーモデルクラスです。"""

    def __init__(self, vocab_size: int, hidden_size: int, dropout_p: float = 0.1, max_len: int = MAX_LENGTH):
        """初期化処理

        Args:
            vocab_size (int): 出力言語（フランス語）のボキャブラリサイズ（重複を除く単語総数）
            hidden_size (int): 隠れ層および分散表現（Embedding）の次元数
            dropout_p (float, optional): ドロップアウト率. Defaults to 0.1.
            max_len (int, optional): 対処する最大シーケンス長. Defaults to MAX_LENGTH.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p
        self.max_len = max_len

        # 各種レイヤーの定義
        self.embed = nn.Embedding(self.vocab_size, self.hidden_size)
        self.dropout = nn.Dropout(p=self.dropout_p)

        # アテンション重み計算用：入力単語ベクトルと現在の隠れ状態（Q）を結合して重みを算出
        self.atten = nn.Linear(self.hidden_size * 2, self.max_len)

        # 結合層：入力単語ベクトルとアテンション適用後のコンテキストベクトルを融合
        self.atten_combin = nn.Linear(self.hidden_size * 2, self.hidden_size)

        # GRU層の定義
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)

        # 出力層（ロジットの計算用と、対数確率に変換するLogSoftmax）
        self.out = nn.Linear(self.hidden_size, self.vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(
            self, input_tensor: torch.Tensor, hidden_tensor: torch.Tensor, encoder_outputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """順伝播処理（フォワードパス）を実行します。

        Args:
            input_tensor (torch.Tensor): 現ステップの入力単語ID [バッチサイズ, 1]
            hidden_tensor (torch.Tensor): 前ステップの隠れ状態（Qに相当） [1, バッチサイズ, 隠れ層次元数]
            encoder_outputs (torch.Tensor): パディング済みのエンコーダー出力（K/Vに相当） [バッチサイズ, 最大シーケンス長, 隠れ層次元数]

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - output_probs (torch.Tensor): 各単語の対数確率分布 [バッチサイズ, ボキャブラリサイズ]
                - hidden (torch.Tensor): 次ステップへ引き継ぐ隠れ状態テンソル [1, バッチサイズ, 隠れ層次元数]
                - attn_weights (torch.Tensor): 算出されたアテンションの重み分布 [バッチサイズ, 最大シーケンス長]
        """
        # [バッチサイズ, 1] -> [バッチサイズ, 1, 隠れ層次元数] への埋め込みおよびドロップアウト適用
        embedded = self.dropout(self.embed(input_tensor))

        # 1. 現在の入力単語とデコーダーの過去隠れ状態を結合してアテンションスコアを計算
        attn_input = torch.cat((embedded.squeeze(1), hidden_tensor[0]), dim=-1)
        attn_weights = F.softmax(self.atten(attn_input), dim=-1)

        # 2. アテンション重みをエンコーダーの出力（Value）に乗算し、コンテキストベクトルを取得
        # batch matrix multiplication (bmm) を使用: [B, 1, max_len] x [B, max_len, H] -> [B, 1, H]
        attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)

        # 3. 入力単語とコンテキストベクトルを結合して特徴量を融合
        combined = torch.cat((embedded.squeeze(1), attn_applied.squeeze(1)), dim=-1)
        gru_input = F.relu(self.atten_combin(combined)).unsqueeze(1)

        # 4. 特徴量をGRUに入力して状態遷移を実行
        output, hidden = self.gru(gru_input, hidden_tensor)

        # 5. 出力層を通じて最終的な単語の予測分布を算出
        output_probs = self.softmax(self.out(output.squeeze(1)))

        return output_probs, hidden, attn_weights

    def init_hidden(self, batch_size: int = 1) -> torch.Tensor:
        """ゼロで初期化された隠れ状態テンソルを生成します。

        Args:
            batch_size (int, optional): ミニバッチのサイズ. Defaults to 1.

        Returns:
            torch.Tensor: 初期化された隠れ状態テンソル [1, バッチサイズ, 隠れ層次元数]
        """
        return torch.zeros(
            1, batch_size, self.hidden_size, device=next(self.parameters()).device
        )


def main() -> None:
    """アテンション付きデコーダーの動作検証用メイン処理です。"""
    # データセットおよびボキャブラリの読み込み
    (
        eng_vocab,
        fra_vocab,
        eng_vocab_size,
        fra_vocab_size,
        fra_idx2word,
        eng_idx2word,
        pairs,
    ) = get_data()

    # データセットおよびデータローダーのインスタンス化
    dataset = TranslationDataset(pairs)
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
    hidden_size = 256

    # 各種ネットワークモデルの構築とデバイスへの転送
    encoder_gru = EncoderGRU(eng_vocab_size, hidden_size).to(device)
    attention_decoder = AttentionDecoder(fra_vocab_size, hidden_size).to(device)
    print(f"アテンション付きデコーダーのモデルアーキテクチャ:\n{attention_decoder}\n")

    # 1ミニバッチ分のみ取り出して挙動を確認（疎通確認）
    for batch_x, batch_y in dataloader:
        print(f"batch_x --> {batch_x}")
        print(f"batch_y --> {batch_y}")

        # デバイス転送
        batch_x = batch_x.to(device)

        # エンコーダーの処理と出力パディングの適用
        h0 = encoder_gru.init_hidden(batch_size=batch_x.size(0))
        encoder_output, hidden = encoder_gru(batch_x, h0)
        encoder_outputs = get_encoder_outputs(encoder_output)
        print(f"encoder_outputs.shape --> {encoder_outputs.shape}")

        # デコーダーの初期状態を設定（開始トークン SOS_TOKEN をバッチサイズ分作成）
        decoder_input = torch.full((batch_x.size(0), 1), SOS_TOKEN, dtype=torch.long, device=device)
        decoder_hidden = hidden

        # タイムステップごとの逐次デコード（ループ処理）の動作検証
        for di in range(batch_y.size(1)):
            output, decoder_hidden, attn_weights = attention_decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            print(f"Step {di}: output.shape={output.shape}, attn_weights.shape={attn_weights.shape}")

            # 最大確率となる単語IDを取得（推論）
            topi = output.argmax(dim=-1, keepdim=True)

            # 終了トークン（EOS_TOKEN）を検出した場合はデコードを終了
            if topi.item() == EOS_TOKEN:
                print("-> EOSトークンを検出したためデコードを終了します。")
                break

            # 次のステップの入力として現在の出力をフィードバック
            decoder_input = topi.detach()
        break


if __name__ == "__main__":
    main()
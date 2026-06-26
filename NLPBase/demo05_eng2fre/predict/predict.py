import os
import sys
from typing import List, Tuple

# 深層学習ライブラリのインポート
import torch

# 自作モジュール用のインポートパス追加設定
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.AttentionDecoder import AttentionDecoder, get_encoder_outputs
from models.EncoderGRU import EncoderGRU
from utils.dataset import EOS_TOKEN, PAD_TOKEN, clean_text, get_data

# 特殊トークンおよび制約条件の定義
MAX_LENGTH = 10  # 扱う文章の最大単語数（記号含む）
SOS_TOKEN = 0  # Start Of Sequence (開始フラグ)
EOS_TOKEN = 1  # End Of Sequence (終了フラグ)

# 実行デバイスの設定（GPUが利用可能であればCUDA、不可であればCPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 学習済みモデルのロード元パス（実行ディレクトリに依存しない絶対パス）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENCODER_PATH = os.path.join(PROJECT_ROOT, "save_model", "ai20_encoder_5.pth")
DECODER_PATH = os.path.join(PROJECT_ROOT, "save_model", "ai20_decoder_5.pth")

# ボキャブラリおよびデータデータの読み込み
(
    eng_vocab,
    fra_vocab,
    eng_vocab_size,
    fra_vocab_size,
    fra_idx2word,
    eng_idx2word,
    _,
) = get_data()


def evaluate_seq2seq(
        input_tensor: torch.Tensor, encoder: EncoderGRU, decoder: AttentionDecoder
) -> Tuple[List[str], torch.Tensor]:
    """入力テンソルに対してSeq2Seqモデルによる推論（翻訳）を実行します。

    Args:
        input_tensor (torch.Tensor): 入力（英語）のIDテンソル [1, シーケンス長]
        encoder (EncoderGRU): 訓練済みのエンコーダーモデル
        decoder (AttentionDecoder): 訓練済みのアテンション付きデコーダーモデル

    Returns:
        Tuple[List[str], torch.Tensor]:
            - decoded_words (List[str]): 翻訳された単語（文字列）のリスト
            - attn_weights (torch.Tensor): 各ステップにおけるアテンション重み行列
    """
    with torch.no_grad():  # 評価フェーズのため勾配計算を無効化
        batch_size = input_tensor.size(0)

        # 1. エンコーダーにデータを入力し、コンテキスト情報を取得
        init_hidden = encoder.init_hidden(batch_size)
        encoder_output, encoder_hidden = encoder(input_tensor, init_hidden)
        encoder_outputs_padded = get_encoder_outputs(encoder_output)

        # デコーダーの初期状態を設定
        decoder_hidden = encoder_hidden
        decoder_input = torch.full((batch_size, 1), SOS_TOKEN, dtype=torch.long, device=device)

        decoded_words = []
        # アテンション重みを記録するテンソルを初期化
        attention_records = torch.zeros(MAX_LENGTH, MAX_LENGTH, device=device)

        # 最大長（MAX_LENGTH）に達するまで1ステップずつデコードを実行
        step_idx = 0
        for step_idx in range(MAX_LENGTH):
            output_probs, decoder_hidden, attn_weight = decoder(
                decoder_input, decoder_hidden, encoder_outputs_padded
            )

            # 最も確率の高い単語とそのインデックスを取得
            _, top_idx = torch.topk(output_probs, k=1, dim=-1)

            # 終了トークン（EOS）が予測された場合は処理を終了
            if top_idx.item() == EOS_TOKEN:
                decoded_words.append("<EOS>")
                break

            # IDを対応するフランス語の単語文字列に変換してリストへ追加
            predicted_word = fra_idx2word[top_idx.item()]
            decoded_words.append(predicted_word)

            # アテンション重みの記録（可視化用データ）
            attention_records[step_idx] = attn_weight[0]

            # 次ステップの予測のために現在の出力を次の入力に設定
            decoder_input = top_idx.detach()

        # 実際に処理したステップ数分のアテンション重みを切り出して返す
        return decoded_words, attention_records[: step_idx + 1]


def test_model() -> None:
    """学習済みモデルをロードし、テストサンプルを用いた翻訳評価テストを実行します。"""
    # 1. エンコーダーの初期化と重みのロード
    encoder_gru = EncoderGRU(eng_vocab_size, hidden_size=256)
    try:
        encoder_gru.load_state_dict(torch.load(ENCODER_PATH, map_location=device))
    except FileNotFoundError:
        print(f"エラー: Encoderのモデルファイルが見つかりません: {ENCODER_PATH}")
        return
    encoder_gru = encoder_gru.to(device)
    encoder_gru.eval()

    # 2. デコーダーの初期化と重みのロード
    attention_decoder = AttentionDecoder(fra_vocab_size, hidden_size=256)
    try:
        attention_decoder.load_state_dict(torch.load(DECODER_PATH, map_location=device))
    except FileNotFoundError:
        print(f"エラー: Decoderのモデルファイルが見つかりません: {DECODER_PATH}")
        return
    attention_decoder = attention_decoder.to(device)
    attention_decoder.eval()

    # テスト検証用の評価サンプルペア [[入力英文, 正解仏文], ...]
    sample_pairs = [
        ["i m impressed with your french .", "je suis impressionne par votre francais ."],
        ["i m more than a friend .", "je suis plus qu une amie ."],
        ["she is beautiful like her mother .", "elle est belle comme sa mere ."],
    ]

    print("=== 推論テストテスト開始 ===")

    # 1サンプルずつ推論処理を実行
    for english_text, target_french in sample_pairs:
        # 入力テキストの単語ID化と末尾へのEOSトークン追加
        source_text = clean_text(english_text)
        input_ids = [eng_vocab.get(word, PAD_TOKEN) for word in source_text.split()]
        input_ids.append(EOS_TOKEN)

        # テンソル化してバッチ次元を追加 [1, シーケンス長] し、デバイスへ転送
        tensor_x = torch.tensor(input_ids, dtype=torch.long, device=device).view(1, -1)

        # 推論（デコード）関数の実行
        predicted_list, _ = evaluate_seq2seq(tensor_x, encoder_gru, attention_decoder)
        predicted_french = " ".join(predicted_list)

        # 評価結果の出力
        print(f"入力英文 (Source) : {english_text}")
        print(f"予測仏文 (Predict): {predicted_french}")
        print(f"正解仏文 (Target) : {target_french}")
        print("-" * 80)


if __name__ == "__main__":
    test_model()

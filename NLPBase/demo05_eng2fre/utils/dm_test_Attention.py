import os
import sys

# 環境変数の設定（重複ライブラリによる警告/エラーの回避）
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib

# 画像保存を安定させるため、PyCharmの独自バックエンドを使わない
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 深層学習ライブラリおよびデータローダーのインポート
import torch

# 自作モジュール用のインポートパス追加設定
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.AttentionDecoder import AttentionDecoder
from models.EncoderGRU import EncoderGRU
from predict.predict import evaluate_seq2seq
from utils.dataset import PAD_TOKEN, clean_text, get_data

# 実行デバイスの設定（GPUが利用可能であればCUDA、不可であればCPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 特殊トークンの定義
SOS_TOKEN = 0  # Start Of Sequence (開始フラグ)
EOS_TOKEN = 1  # End Of Sequence (終了フラグ)
PAD_TOKEN = 2  # Padding (パディングフラグ)

# 学習済みモデルのロード元パス
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENCODER_PATH = os.path.join(PROJECT_ROOT, "save_model", "ai20_encoder_5.pth")
DECODER_PATH = os.path.join(PROJECT_ROOT, "save_model", "ai20_decoder_5.pth")
OUTPUT_IMAGE_PATH = os.path.join(PROJECT_ROOT, "img", "s2s_attn.png")


def plot_attention_map() -> None:
    """学習済みモデルを用いてアテンション（注視度）の可視化テストを実行します。

    テスト用データをモデルに入力し、出力されたアテンション行列をグラフとして描画・保存します。
    """
    # データセットおよびボキャブラリの読み込み
    (
        eng_vocab,
        fra_vocab,
        eng_vocab_size,
        fra_vocab_size,
        fra_idx2word,
        eng_idx2word,
        _,
    ) = get_data()

    # 共通ハイパーパラメータの設定
    hidden_size = 256  # 隠れ層の次元数（デバッグ時は8などの小サイズも検討可）

    # 1. Encoder（符号化器）のインスタンス化と重みのロード
    encoder = EncoderGRU(eng_vocab_size, hidden_size)
    try:
        encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=device))
    except FileNotFoundError:
        print(f"エラー: Encoderのモデルファイルが見つかりません: {ENCODER_PATH}")
        return
    encoder = encoder.to(device)
    encoder.eval()  # 推論モードに設定

    # 2. Decoder（復号化器）のインスタンス化と重みのロード
    decoder = AttentionDecoder(fra_vocab_size, hidden_size)
    try:
        decoder.load_state_dict(torch.load(DECODER_PATH, map_location=device))
    except FileNotFoundError:
        print(f"エラー: Decoderのモデルファイルが見つかりません: {DECODER_PATH}")
        return
    decoder = decoder.to(device)
    decoder.eval()  # 推論モードに設定

    # テスト対象の入力文章定義
    sample_sentence = "we re both teachers ."

    # 入力テキストの単語をIDに変換し、末尾にEOSトークンを付与
    sample_sentence = clean_text(sample_sentence)
    input_ids = [eng_vocab.get(word, PAD_TOKEN) for word in sample_sentence.split()]
    input_ids.append(EOS_TOKEN)

    # モデル入力用にテンソル化（形状: [1, シーケンス長]）し、指定デバイスへ転送
    tensor_x = torch.tensor(input_ids, dtype=torch.long, device=device).view(1, -1)

    # モデルによる推論実行（翻訳結果およびアテンション重みの取得）
    with torch.no_grad():  # 勾配計算を無効化してメモリを節約
        decoded_words, attentions = evaluate_seq2seq(tensor_x, encoder, decoder)

    print(f"decoded_words -> {decoded_words}")

    # アテンション重みを可視化用にNumPy配列に変換
    attentions_np = attentions.detach().cpu().numpy()

    # マトリックス形式でプロットを表示
    plt.matshow(attentions_np)

    # グラフの保存
    os.makedirs(os.path.dirname(OUTPUT_IMAGE_PATH), exist_ok=True)
    plt.title(f"Attention: {sample_sentence}")
    plt.xlabel("Input position")
    plt.ylabel("Output position")
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE_PATH, dpi=150)
    plt.close()
    print(f"アテンション画像を保存しました: {OUTPUT_IMAGE_PATH}")


if __name__ == "__main__":
    plot_attention_map()

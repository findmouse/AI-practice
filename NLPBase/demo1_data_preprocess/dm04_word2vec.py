import os
import fasttext
import numpy as np

# 定数定義（マジックナンバーやハードコーディングの回避）
TRAIN_DATA_PATH = "../assets/ai20aa"
MODEL_OUTPUT_PATH = "../assets/ai20_fil9.bin"


def train_unsupervised_model() -> None:
    """
    教師なし学習（Unsupervised Learning）によるモデルの訓練と保存を行います。
    """
    print("モデルの訓練を開始します...")

    # 事前チェック: 訓練データの存在確認
    if not os.path.exists(TRAIN_DATA_PATH):
        raise FileNotFoundError(f"訓練データが見つかりません: {TRAIN_DATA_PATH}")

    # 教師なし学習モードでモデルを訓練
    model = fasttext.train_unsupervised(TRAIN_DATA_PATH, "cbow", dim=300, epoch=1, lr=0.1, thread=8)

    # 訓練済みモデルをローカル環境に保存
    model.save_model(MODEL_OUTPUT_PATH)
    print(f"モデルの保存が完了しました: {MODEL_OUTPUT_PATH}")


def verify_word_vectors() -> None:
    """
    保存されたモデルをロードし、単語ベクトルの取得および類似度の検証を行います。
    """
    # 事前チェック: モデルファイルの存在確認
    if not os.path.exists(MODEL_OUTPUT_PATH):
        raise FileNotFoundError(f"モデルファイルが見つかりません: {MODEL_OUTPUT_PATH}")

    print(f"モデルをロードしています: {MODEL_OUTPUT_PATH}")
    model = fasttext.load_model(MODEL_OUTPUT_PATH)

    # 特定の単語（例: 'the'）のベクトルを取得
    target_word = "the"
    word_vector: np.ndarray = model.get_word_vector(target_word)

    print("\n--- 単語ベクトルの情報 ---")
    print(f"データ型: {type(word_vector)}")
    print(f"ベクトルの次元数（Shape）: {word_vector.shape}")

    # モデルの精度・効果の検証（類似度の高い近傍語の取得）
    print("\n--- 類似単語（Nearest Neighbors）の検証 ---")
    test_words = ["the", "music", "hotdog"]
    for word in test_words:
        print(f"\n【{word}】の類似単語:")
        neighbors = model.get_nearest_neighbors(word)
        for score, neighbor_word in neighbors:
            print(f"  - {neighbor_word} (スコア: {score:.4f})")


if __name__ == '__main__':
    # 実行したい処理のコメントアウトを解除して実行してください
    train_unsupervised_model()
    verify_word_vectors()

import jieba
import joblib
import numpy as np
import torch
import torch.nn as nn
from tensorflow.keras.preprocessing.text import Tokenizer
from torch.utils.tensorboard import SummaryWriter

# 定数定義
EMBEDDING_DIM = 8  # 埋め込みベクトルの次元数
LOG_DIR = "./runs"  # TensorBoardのログ出力先


def execute_embedding_pipeline() -> None:
    """
    中国語のテキストを形態素解析し、トークン化、
    およびPyTorchのEmbedding層を用いた分散表現（単語ベクトル）の生成と検証を行います。
    """
    # 1. サンプルテキストの定義と形態素解析（分詞処理）
    sentence1 = "传智教育是一家上市公司，旗下有黑马程序员品牌。我是在黑马这里学习人工智能"
    sentence2 = "我爱自然语言处理"
    sentences = [sentence1, sentence2]

    word_list = []
    for s in sentences:
        # jiebaを用いて分詞（中国語の単語区切り）を実行
        word_list.append(jieba.lcut(s))

    print(f"【分詞結果】: {word_list}")

    # 2. Tokenizerを用いた単語辞書の作成
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(word_list)

    # KerasのTokenizerは1インデックスのため、総単語数は「辞書の長さ + 1」とする（0番目はパディング用）
    vocab_size = len(tokenizer.word_index) + 1
    token_labels = [tokenizer.index_word[i] for i in range(1, vocab_size)]

    # テキストをインデックスのシーケンスに変換
    seq2id = tokenizer.texts_to_sequences(word_list)
    print(f"【インデックス変換（ID化）】: {seq2id}")

    # 3. PyTorchのEmbedding層の生成
    # num_embeddingsにはパディングを考慮したvocab_sizeを指定
    embed_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=EMBEDDING_DIM)

    print("\n--- Embedding層の情報 ---")
    print(f"重み行列の形状 (Shape): {embed_layer.weight.data.shape}")

    # 4. TensorBoardによる可視化（オプショナル）
    # ※ 視覚的に確認したい場合は、以下のコメントアウトを解除してください。
    # ※ 実行後、ターミナルで「tensorboard --logdir=runs」を実行します。
    # print(f"\nTensorBoardのログを保存中: {LOG_DIR}")
    # with SummaryWriter(log_dir=LOG_DIR) as writer:
    #     # 1から始まる有効な単語ベクトルのみを抽出して可視化に渡す
    #     valid_weights = embed_layer.weight.data[1:]
    #     writer.add_embedding(valid_weights, metadata=token_labels)

    # 5. 各単語に対応する埋め込みベクトル（分散表現）の抽出と表示
    print("\n--- 各単語の埋め込みベクトル一覧 ---")
    for idx, word in tokenizer.index_word.items():
        # テンソル型に変換してEmbedding層に入力
        input_tensor = torch.tensor([idx], dtype=torch.long)
        word_vector = embed_layer(input_tensor)

        # 扱いやすいようにNumPy配列に変換（1次元に平坦化）
        vector_numpy = word_vector.detach().numpy().squeeze()

        print(f"単語: 【{word}】(ID: {idx})")
        print(f"  -> ベクトル: {vector_numpy}")


if __name__ == '__main__':
    execute_embedding_pipeline()

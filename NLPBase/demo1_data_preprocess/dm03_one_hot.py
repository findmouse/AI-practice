# coding: utf-8

"""
Tokenizer を使用した One-Hot エンコーディングのサンプル

対象データ:
    周杰伦
    陈奕迅
    王力宏
    李宗盛
    吴亦凡
    鹿晗
"""

from tensorflow.keras.preprocessing.text import Tokenizer
import joblib

# Tokenizer 保存先
TOKENIZER_PATH = "./mytokenizer"

# 学習対象の語彙
VOCABS = {
    "周杰伦",
    "陈奕迅",
    "王力宏",
    "李宗盛",
    "吴亦凡",
    "鹿晗"
}


def get_one_hot():
    """
    Tokenizer を生成し、
    各単語の One-Hot ベクトルを表示する。

    また、作成した Tokenizer をファイルへ保存する。
    """

    # Tokenizer を生成
    tokenizer = Tokenizer()

    # 語彙を学習
    tokenizer.fit_on_texts(VOCABS)

    print("word_index:")
    print(tokenizer.word_index)

    print("index_word:")
    print(tokenizer.index_word)

    # 各単語の One-Hot ベクトルを生成
    for vocab in VOCABS:
        one_hot = [0] * len(VOCABS)

        index = tokenizer.word_index[vocab] - 1
        one_hot[index] = 1

        print(f"現在の {vocab} の One-Hot エンコード: {one_hot}")

    # Tokenizer を保存
    joblib.dump(tokenizer, TOKENIZER_PATH)

    print("Tokenizer の保存が完了しました。")


def use_one_hot():
    """
    保存済みの Tokenizer を読み込み、
    指定単語の One-Hot ベクトルを生成する。
    """

    token = "李宗盛"

    # Tokenizer をロード
    tokenizer = joblib.load(TOKENIZER_PATH)

    print("word_index:")
    print(tokenizer.word_index)

    one_hot = [0] * len(VOCABS)

    index = tokenizer.word_index[token] - 1
    one_hot[index] = 1

    print(one_hot)
    print(f"現在の {token} の One-Hot エンコード: {one_hot}")


if __name__ == "__main__":
    # Tokenizer 作成・保存
    # get_one_hot()

    # 保存済み Tokenizer の利用
    use_one_hot()

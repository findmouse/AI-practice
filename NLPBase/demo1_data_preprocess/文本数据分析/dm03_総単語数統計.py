import os
from typing import Set
import jieba
import pandas as pd

# 定数定義
TRAIN_DATA_PATH = "../../assets/train.tsv"
DEV_DATA_PATH = "../../assets/dev.tsv"


def extract_vocabulary_from_tsv(file_path: str) -> Set[str]:
    """
    指定されたTSVファイルからテキストデータを読み込み、
    形態素解析（分詞処理）を行って一意な単語の集合（語彙）を抽出します。

    Args:
        file_path (str): 読み込み対象のTSVファイルパス

    Returns:
        Set[str]: 重複のない単語の集合（語彙セット）
    """
    # 事前チェック: ファイルの存在確認
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"データファイルが見つかりません: {file_path}")

    # データの読み込み
    df = pd.read_csv(filepath_or_buffer=file_path, sep="\t")

    # 集合内包表記（Set Comprehension）を用いて、重複のない語彙を効率的に抽出
    # ※ chain + map よりも、Pythonでは内包表記の方が可読性が高く好まれる傾向にあります
    vocabulary: Set[str] = {
        word
        for sentence in df["sentence"]
        for word in jieba.lcut(sentence)
    }

    return vocabulary


def main() -> None:
    """
    メイン処理: 訓練データと検証データの語彙数を集計し、コンソールに出力します。
    """
    print("各データの語彙抽出を開始します...")

    # 訓練データおよび検証データから語彙を抽出
    train_vocabs = extract_vocabulary_from_tsv(TRAIN_DATA_PATH)
    dev_vocabs = extract_vocabulary_from_tsv(DEV_DATA_PATH)

    # 集計結果の出力
    print("\n--- 語彙数（Vocabulary Size）集計結果 ---")
    print(f"訓練データ（Train）の一意な単語数 : {len(train_vocabs)}")
    print(f"検証データ（Dev）の一意な単語数   : {len(dev_vocabs)}")

    # 実務的なTips: 訓練データと検証データの語彙の重複度合い（カバー率）を確認するコード
    # （未知語/OOVの発生リスクを把握するために、日本の現場ではよく合わせて確認されます）
    oov_words = dev_vocabs - train_vocabs
    print(f"検証データにのみ含まれる未知語数   : {len(oov_words)}")


if __name__ == "__main__":
    main()
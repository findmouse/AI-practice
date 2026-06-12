import os
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# GUIバックエンドの設定
matplotlib.use("TkAgg")

# 定数定義
TRAIN_DATA_PATH = "../../assets/train.tsv"
DEV_DATA_PATH = "../../assets/dev.tsv"
PLOT_STYLE = "fivethirtyeight"


def load_and_preprocess_data(file_path: str) -> pd.DataFrame:
    """
    TSVファイルを読み込み、文章の長さ（文字数）を新規カラムとして追加します。

    Args:
        file_path (str): 読み込み対象のファイルパス

    Returns:
        pd.DataFrame: 'sentence_length' カラムが追加されたデータフレーム
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"データファイルが見つかりません: {file_path}")

    df = pd.read_csv(filepath_or_buffer=file_path, sep="\t")

    # 補足: lambdaを使うよりも、Pandasの拡張プロパティ（.str.len()）を使う方が高速かつシンプルです
    df["sentence_length"] = df["sentence"].str.len()

    return df


def plot_text_length_distribution(data: pd.DataFrame, title: str) -> None:
    """
    文章長さの分布を、カウントプロットおよび分布図（ヒストグラム/曲線）として可視化します。
    """
    plt.style.use(PLOT_STYLE)

    # 1. カウントプロット（柱状図）の描画
    plt.figure()
    sns.countplot(x="sentence_length", data=data)
    plt.title(f"{title} - Count Plot")
    plt.tight_layout()
    plt.show()

    # 2. 分布図（曲線図 / ヒストグラム）の描画
    # 注: displotはFigureレベルの関数のため、plt.figure()ではなく直接呼び出し、戻り値に対してタイトルを設定します
    g = sns.displot(x="sentence_length", data=data, kde=True)
    g.set_titles(f"{title} - Distribution")
    plt.show()


def plot_length_stripplot(data: pd.DataFrame, title: str) -> None:
    """
    正例・負例（ラベルごと）の文章長さの散布分布（ストリッププロット）を描画します。
    """
    plt.style.use(PLOT_STYLE)

    plt.figure()
    # 独自のカラーパレットやデータの重なりを防ぐ設定（jitter）を明記すると実務的です
    sns.stripplot(
        y="sentence_length",
        x="label",
        data=data,
        hue="label",
        jitter=True,
        palette="Set2"
    )
    plt.title(f"{title} - Label vs Length")
    plt.tight_layout()
    plt.show()


def main() -> None:
    """
    メイン処理: 訓練データと検証データの可視化パイプラインを実行します。
    """
    print("データの読み込みおよび前処理を開始します...")
    train_df = load_and_preprocess_data(TRAIN_DATA_PATH)
    dev_df = load_and_preprocess_data(DEV_DATA_PATH)

    # データ確認用のログ出力
    print(f"\n--- 訓練データ（先頭5行） ---\n{train_df.head()}")
    print(f"\n--- 検証データ（先頭5行） ---\n{dev_df.head()}")

    # 1. 文章長さの分布可視化（コメントアウトを解除して実行可能）
    # print("\n文章長さの分布を描画中...")
    # plot_text_length_distribution(train_df, "Train Data")
    # plot_text_length_distribution(dev_df, "Dev Data")

    # 2. ラベルごとの散布分布可視化
    print("\nラベルごとの散布分布を描画中...")
    plot_length_stripplot(train_df, "Train Data")
    plot_length_stripplot(dev_df, "Dev Data")


if __name__ == "__main__":
    main()
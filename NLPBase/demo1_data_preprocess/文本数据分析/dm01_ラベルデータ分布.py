import os
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# GUIバックエンドの設定（環境に応じて調整してください）
matplotlib.use("TkAgg")

# 定数定義
TRAIN_DATA_PATH = "../../assets/train.tsv"
DEV_DATA_PATH = "../../assets/dev.tsv"
PLOT_STYLE = "fivethirtyeight"


def plot_label_distribution(data: pd.DataFrame, title: str) -> None:
    """
    指定されたDataFrameの'label'列の分布をカウントプロットとして描画・表示します。

    Args:
        data (pd.DataFrame): 対象のデータフレーム
        title (str): グラフのタイトル
    """
    # グラフのスタイルを適用
    plt.style.use(PLOT_STYLE)

    # 新しいウィンドウ（フィギュア）を作成
    plt.figure()

    # seabornを使用してラベル（0, 1など）のグループ別数量をカウント
    sns.countplot(x="label", data=data)

    # タイトルの設定と描画の実行
    plt.title(title)
    plt.tight_layout()  # レイアウトの自動調整（文字切れ防止）
    plt.show()


def analyze_label_distributions() -> None:
    """
    訓練データおよび検証データを読み込み、それぞれのラベル分布を可視化します。
    """
    # 事前チェック: ファイルの存在確認
    for path in [TRAIN_DATA_PATH, DEV_DATA_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"データファイルが見つかりません: {path}")

    print("データを読み込んでいます...")
    # TSVファイル（タブ区切り）の読み込み
    train_data = pd.read_csv(filepath_or_buffer=TRAIN_DATA_PATH, sep="\t")
    dev_data = pd.read_csv(filepath_or_buffer=DEV_DATA_PATH, sep="\t")

    # 訓練データのラベル分布を出力
    print("訓練データのラベル分布を描画中...")
    plot_label_distribution(data=train_data, title="Train Label Distribution")

    # 検証データのラベル分布を出力
    print("検証データのラベル分布を描画中...")
    plot_label_distribution(data=dev_data, title="Dev Label Distribution")


if __name__ == "__main__":
    analyze_label_distributions()

# coding:utf-8
import logging
import os
import sys

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# 環境変数の設定（重複ライブラリによる警告/エラーの回避）
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 独自のユーティリティモジュールのパス追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.dataset import read_json

# ロギングの設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def dm_show_results() -> None:
    """
    RNN、LSTM、GRUの学習結果（損失、実行時間、精度）を比較し、
    グラフを描画して画像として保存します。
    """
    # 保存先ディレクトリの作成（存在しない場合）
    os.makedirs('../img', exist_ok=True)

    # 各モデルの学習結果をJSONファイルから読み込み
    try:
        rnn_avg_loss, rnn_all_time, rnn_avg_acc = read_json("../save_results/ai_rnn_train_result.json")
        lstm_avg_loss, lstm_all_time, lstm_avg_acc = read_json("../save_results/ai_lstm_train_result.json")
        gru_avg_loss, gru_all_time, gru_avg_acc = read_json("../save_results/ai_gru_train_result.json")
    except FileNotFoundError as e:
        logging.error(f"結果のJSONファイルが見つかりません: {e}")
        return

    # 各モデルの実行時間のログ出力
    logging.info(f"RNN 総実行時間: {rnn_all_time}")
    logging.info(f"LSTM 総実行時間: {lstm_all_time}")
    logging.info(f"GRU 総実行時間: {gru_all_time}")
    logging.info(f"RNN 平均精度リスト（一部）: {rnn_avg_acc[:5] if isinstance(rnn_avg_acc, list) else rnn_avg_acc}")

    # 1. 各モデルの損失（Loss）の比較グラフ
    plt.figure(0)
    plt.plot(rnn_avg_loss, label="RNN")
    plt.plot(lstm_avg_loss, label="LSTM", color='red')
    plt.plot(gru_avg_loss, label="GRU", color='orange')
    plt.title("Model Training Loss Comparison")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(loc='upper right')
    plt.savefig('../img/loss.png')
    logging.info("損失比較グラフを '../img/loss.png' に保存しました。")
    plt.show()

    # 2. 各モデルの総実行時間（Time）の比較グラフ
    plt.figure(1)
    x_data = ["RNN", "LSTM", "GRU"]
    y_data = [rnn_all_time, lstm_all_time, gru_all_time]
    plt.bar(range(len(x_data)), y_data, tick_label=x_data, color=['blue', 'red', 'orange'])
    plt.title("Model Execution Time Comparison")
    plt.ylabel("Time (seconds)")
    plt.savefig('../img/time.png')
    logging.info("実行時間比較グラフを '../img/time.png' に保存しました。")
    plt.show()

    # 3. 各モデルの正解率（Accuracy）の比較グラフ
    plt.figure(2)
    plt.plot(rnn_avg_acc, label="RNN")
    plt.plot(lstm_avg_acc, label="LSTM", color='red')
    plt.plot(gru_avg_acc, label="GRU", color='orange')
    plt.title("Model Accuracy Comparison")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(loc='lower right')
    plt.savefig('../img/acc.png')
    logging.info("正解率比較グラフを '../img/acc.png' に保存しました。")
    plt.show()


if __name__ == '__main__':
    dm_show_results()
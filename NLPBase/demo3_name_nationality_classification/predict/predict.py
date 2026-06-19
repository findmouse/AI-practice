# coding:utf-8
import logging
import os
import string
import sys

import torch

# 独自のユーティリティモジュールのパス追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.rnn_model import My_RNN
from models.lstm_model import My_LSTM
from models.gru_model import My_GRU

# ロギングの設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 利用可能な文字セット（アルファベット + 記号）
letters = string.ascii_letters + " ,.;'"
n_letters = len(letters)

# 分類対象の国籍リスト
categories = [
    'Italian', 'English', 'Arabic', 'Spanish', 'Scottish', 'Irish',
    'Chinese', 'Vietnamese', 'Japanese', 'French', 'Greek', 'Dutch',
    'Korean', 'Polish', 'Portuguese', 'Russian', 'Czech', 'German'
]

# モデルの永続化ファイルパス
RNN_MODEL_PATH = "../save_model/my_rnn_model_1.bin"
LSTM_MODEL_PATH = "../save_model/my_lstm_model_1.bin"
GRU_MODEL_PATH = "../save_model/my_gru_model_1.bin"  # 既存のタイポ（rnn_model_1）を修正しました


def line_to_tensor(x: str) -> torch.Tensor:
    """
    文字列（名前）をOne-hot表現のテンソルに変換します。
    """
    tensor_x = torch.zeros(len(x), n_letters)
    for li, letter in enumerate(x):
        letter_pos = letters.find(letter)
        if letter_pos != -1:
            tensor_x[li][letter_pos] = 1.0
    return tensor_x


def rnn_predict(x: str) -> None:
    """
    RNNモデルを使用して国籍を推論し、上位3つの予測結果を表示します。
    """
    if not os.path.exists(RNN_MODEL_PATH):
        logging.warning(f"RNNモデルファイルが見つかりません: {RNN_MODEL_PATH}")
        return

    tensor_x = line_to_tensor(x)
    my_rnn = My_RNN(input_size=57, hidden_size=128, output_size=18)
    my_rnn.load_state_dict(torch.load(RNN_MODEL_PATH))
    my_rnn.eval()

    with torch.no_grad():
        output, hn = my_rnn(tensor_x, my_rnn.init_hidden())
        values, indexes = torch.topk(output, k=3)

        logging.info(f"[RNN] 推論開始（対象: {x}）")
        for i in range(3):
            value = values[0][i].item()
            index = indexes[0][i].item()  # インデックスのバグ（[0][1]固定になっていた点）を修正
            category = categories[index]
            print(f"予測順位 {i + 1}: 確率(対数ハザード): {value:.4f} | 予測結果: {category}")


def lstm_predict(x: str) -> None:
    """
    LSTMモデルを使用して国籍を推論し、上位3つの予測結果を表示します。
    """
    if not os.path.exists(LSTM_MODEL_PATH):
        logging.warning(f"LSTMモデルファイルが見つかりません: {LSTM_MODEL_PATH}")
        return

    tensor_x = line_to_tensor(x)
    my_lstm = My_LSTM(input_size=57, hidden_size=128, output_size=18)
    my_lstm.load_state_dict(torch.load(LSTM_MODEL_PATH))
    my_lstm.eval()

    with torch.no_grad():
        hidden, c = my_lstm.init_hidden()
        output, hn, cn = my_lstm(tensor_x, hidden, c)
        values, indexes = torch.topk(output, k=3)

        logging.info(f"[LSTM] 推論開始（対象: {x}）")
        for i in range(3):
            value = values[0][i].item()
            index = indexes[0][i].item()  # インデックスのバグ（[0][1]固定になっていた点）を修正
            category = categories[index]
            print(f"予測順位 {i + 1}: 確率(対数ハザード): {value:.4f} | 予測結果: {category}")


def gru_predict(x: str) -> None:
    """
    GRUモデルを使用して国籍を推論し、上位3つの予測結果を表示します。
    """
    if not os.path.exists(GRU_MODEL_PATH):
        logging.warning(f"GRUモデルファイルが見つかりません: {GRU_MODEL_PATH}")
        return

    tensor_x = line_to_tensor(x)
    my_gru = My_GRU(input_size=57, hidden_size=128, output_size=18)  # My_RNNからMy_GRUへ修正
    my_gru.load_state_dict(torch.load(GRU_MODEL_PATH))
    my_gru.eval()

    with torch.no_grad():
        output, hn = my_gru(tensor_x, my_gru.init_hidden())
        values, indexes = torch.topk(output, k=3)

        logging.info(f"[GRU] 推論開始（対象: {x}）")
        for i in range(3):
            value = values[0][i].item()
            index = indexes[0][i].item()  # インデックスのバグ（[0][1]固定になっていた点）を修正
            category = categories[index]
            print(f"予測順位 {i + 1}: 確率(対数ハザード): {value:.4f} | 予測結果: {category}")


if __name__ == '__main__':
    target_name = "addddma"

    rnn_predict(target_name)
    print("=" * 50)
    lstm_predict(target_name)
    print("=" * 50)
    gru_predict(target_name)
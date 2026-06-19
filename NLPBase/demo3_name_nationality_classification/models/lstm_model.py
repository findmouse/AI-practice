# coding:utf-8
import logging
import os
import sys
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

# 独自のユーティリティモジュールのパス追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.NameClassDataset import NameClassDataset
from utils.dataset import read_data

# ロギングの設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class My_LSTM(nn.Module):
    """
    名前（文字列）を分類するためのLSTMカスタムモデルクラスです。
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layer: int = 1) -> None:
        super().__init__()
        # 文字の埋め込み次元数（入力サイズ）
        self.input_size = input_size
        # LSTMモデルが出力する隠れ層の次元数
        self.hidden_size = hidden_size
        # 最終出力層のユニット数（クラス数）
        self.output_size = output_size
        # LSTMのレイヤー数
        self.num_layer = num_layer

        # LSTMレイヤーの定義
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layer)
        # 全結合層の定義
        self.linear = nn.Linear(self.hidden_size, self.output_size)
        # Softmax層の定義（最後の次元に対して計算）
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input_tensor: torch.Tensor, hidden: torch.Tensor, cell: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        順伝播処理を行います。

        Args:
            input_tensor (torch.Tensor): 入力テンソル。形状は [シーケンス長, 入力次元数]
            hidden (torch.Tensor): 初期隠れ状態テンソル (h0)
            cell (torch.Tensor): 初期セル状態テンソル (c0)

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 出力結果、更新された隠れ状態、更新されたセル状態
        """
        # input_tensorの形状を [シーケンス長, バッチサイズ(1), 入力次元数] に変換
        input_tensor = input_tensor.unsqueeze(1)

        # LSTMレイヤーへ入力（隠れ状態とセル状態をタプルで渡す）
        output, (hn, cn) = self.lstm(input_tensor, (hidden, cell))

        # シーケンスの最後の出力（output[-1]）を全結合層へ入力
        result = self.linear(output[-1])

        # LogSoftmaxを適用して返却
        return self.softmax(result), hn, cn

    def init_hidden(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        初期の隠れ状態(h0)とセル状態(c0)をゼロテンソルとして生成します。
        """
        h0 = torch.zeros(self.num_layer, 1, self.hidden_size)
        c0 = torch.zeros(self.num_layer, 1, self.hidden_size)
        return h0, c0


# モデル動作確認用メイン処理
if __name__ == '__main__':
    # モデルのインスタンス化
    my_lstm = My_LSTM(input_size=57, hidden_size=128, output_size=18)

    # データの読み込みとデータローダーの初期化
    data_path = '../data/name_classfication.txt'
    if not os.path.exists(data_path):
        logging.error(f"データファイルが見つかりません: {data_path}")
        sys.exit(1)

    my_list_x, my_list_y = read_data(data_path)
    my_dataset = NameClassDataset(my_list_x, my_list_y)
    my_dataloader = DataLoader(dataset=my_dataset, batch_size=1, shuffle=True)

    logging.info(f"データローダーの総バッチ数 (len): {len(my_dataloader)}")

    # 最初の1バッチのみを取り出して動作確認
    for tensor_x, tensor_y in my_dataloader:
        # batch_size=1 のため、最初のサンプルを取得
        sample_input = tensor_x[0]

        # 初期状態の生成と順伝播の実行
        initial_hidden, initial_cell = my_lstm.init_hidden()
        output_tensor, hidden_out, cell_out = my_lstm(sample_input, initial_hidden, initial_cell)

        # 結果の出力
        logging.info(f"出力テンソルの形状 (output.shape): {output_tensor.shape}")
        logging.info(f"隠れ状態テンソルの形状 (hn.shape): {hidden_out.shape}")
        logging.info(f"セル状態テンソルの形状 (cn.shape): {cell_out.shape}")
        break
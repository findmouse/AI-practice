# coding:utf-8
import json
import logging
import os
import sys
import time
from typing import Tuple, List

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# 独自のユーティリティモジュールのパス追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.NameClassDataset import NameClassDataset
from utils.dataset import read_data
from models.rnn_model import My_RNN

# ロギングの設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ハイパーパラメータの設定
LR = 1e-3
EPOCHS = 1


def train_rnn() -> Tuple[List[float], int, List[float]]:
    """
    RNNモデルの学習を行い、損失の履歴、総実行時間、精度の履歴を返します。
    また、結果をJSONファイル、モデルをバイナリファイルとして保存します。
    """
    # データの読み込み
    data_path = '../data/name_classfication.txt'
    if not os.path.exists(data_path):
        logging.error(f"データファイルが見つかりません: {data_path}")
        sys.exit(1)

    my_list_x, my_list_y = read_data(data_path)
    my_dataset = NameClassDataset(my_list_x, my_list_y)

    # モデル、損失関数、最適化アルゴリズムの初期化
    # n_letters=57, hidden_size=128, output_size=18
    my_rnn = My_RNN(input_size=57, hidden_size=128, output_size=18)
    my_nll_loss = nn.NLLLoss()
    my_optim = optim.Adam(my_rnn.parameters(), lr=LR)

    # メトリクス記録用変数の初期化
    start_time = time.time()
    total_iter_num = 0  # 学習済みの総サンプル数
    total_loss = 0.0  # 累積損失
    total_loss_list = []  # 一定間隔ごとの平均損失履歴
    total_acc_num = 0  # 正解サンプル数
    total_acc_list = []  # 一定間隔ごとの平均精度履歴

    logging.info("RNNモデルの学習を開始します。")

    # 学習ループ
    for epoch_idx in range(EPOCHS):
        my_dataloader = DataLoader(dataset=my_dataset, batch_size=1, shuffle=True)

        for i, (x, y) in enumerate(tqdm(my_dataloader, desc=f"Epoch {epoch_idx + 1}/{EPOCHS}")):
            # 順伝播処理（batch_size=1のためx[0]を抽出）
            output, hn = my_rnn(input_tensor=x[0], hidden=my_rnn.init_hidden())

            # 損失の計算
            my_loss = my_nll_loss(output, y)

            # 勾配の初期化、逆伝播、パラメータ更新
            my_optim.zero_grad()
            my_loss.backward()
            my_optim.step()

            # メトリクスの更新
            total_iter_num += 1
            total_loss += my_loss.item()

            # 正解数のカウント
            is_correct = 1 if torch.argmax(output).item() == y.item() else 0
            total_acc_num += is_correct

            # 100イテレーションごとに平均損失と平均精度を記録
            if total_iter_num % 100 == 0:
                avg_loss = total_loss / total_iter_num
                total_loss_list.append(avg_loss)

                avg_acc = total_acc_num / total_iter_num
                total_acc_list.append(avg_acc)

            # 2000イテレーションごとに進捗ログを出力
            if total_iter_num % 2000 == 0:
                temp_loss = total_loss / total_iter_num
                temp_acc = total_acc_num / total_iter_num
                temp_time = time.time() - start_time
                logging.info(
                    f"Epoch: {epoch_idx + 1} | Iter: {total_iter_num} | "
                    f"Loss: {temp_loss:.6f} | Time: {int(temp_time)}s | Acc: {temp_acc:.3f}"
                )

        # エポックごとにモデルを保存
        model_dir = '../save_model'
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f'my_rnn_model_{epoch_idx + 1}.bin')
        torch.save(my_rnn.state_dict(), model_path)
        logging.info(f"モデルを保存しました: {model_path}")

    # 総実行時間の計算
    total_time = int(time.time() - start_time)
    logging.info(f"学習完了。総実行時間: {total_time}秒")

    # 結果の保存処理
    save_data = {
        "loss": total_loss_list,
        "time": total_time,
        "acc": total_acc_list
    }

    results_dir = '../save_results'
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, 'ai_rnn_train_result.json')

    with open(results_path, 'w', encoding='utf-8') as fw:
        json.dump(save_data, fw, ensure_ascii=False, indent=4)
    logging.info(f"学習結果を保存しました: {results_path}")

    return total_loss_list, total_time, total_acc_list


if __name__ == '__main__':
    train_rnn()
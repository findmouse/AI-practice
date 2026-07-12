# coding:utf-8
import json
import logging
import os
import sys
from typing import Tuple, List

from torch.utils.data import DataLoader

# 自作モジュールのインポートパス設定
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from NameClassDataset import NameClassDataset

# ロギングの設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def read_data(filename: str) -> Tuple[List[str], List[str]]:
    """
    ファイルからテキストデータを読み込み、入力データとラベルのリストを返します。

    Args:
        filename (str): 読み込み対象のファイルパス

    Returns:
        Tuple[List[str], List[str]]: 入力データリスト(my_list_x)とラベルリスト(my_list_y)
    """
    my_list_x, my_list_y = [], []

    if not os.path.exists(filename):
        logging.error(f"ファイルが見つかりません: {filename}")
        return my_list_x, my_list_y

    with open(filename, mode='r', encoding='utf-8') as f:
        for line in f:
            # 5文字以下の無効な行はスキップ
            if len(line) <= 5:
                continue

            # タブ区切りで入力データ(x)とラベル(y)を抽出
            try:
                x, y = line.strip().split('\t')
                my_list_x.append(x)
                my_list_y.append(y)
            except ValueError:
                logging.warning(f"不正なフォーマットの行をスキップしました: {line.strip()}")
                continue

    return my_list_x, my_list_y


def get_dataloader() -> None:
    """
    データローダーを作成し、最初のバッチのデータ形状を確認します。
    """
    data_path = '../data/name_classfication.txt'
    my_list_x, my_list_y = read_data(data_path)

    my_dataset = NameClassDataset(my_list_x, my_list_y)
    my_dataloader = DataLoader(dataset=my_dataset, batch_size=1, shuffle=True)
    #
    # logging.info(f"データローダーの総バッチ数 (len): {len(my_dataloader)}")
    #
    # # 最初の1バッチのみ確認
    # for tensor_x, tensor_y in my_dataloader:
    #     logging.info(f"入力テンソルの形状 (tensor_x.shape): {tensor_x.shape}")
    #     logging.info(f"正解ラベル (tensor_y): {tensor_y}")
    #     break

    return my_dataloader


def read_json(data_path: str) -> Tuple[float, float, float]:
    """
    JSONファイルから学習結果（損失、時間、精度）を読み込みます。

    Args:
        data_path (str): JSONファイルのパス

    Returns:
        Tuple[float, float, float]: 平均損失、総実行時間、平均精度
    """
    with open(data_path, 'r', encoding='utf-8') as fr:
        results = json.load(fr)

    avg_loss = results.get("loss", 0.0)
    all_time = results.get("time", 0.0)
    avg_acc = results.get("acc", 0.0)

    return avg_loss, all_time, avg_acc


# def test_dataset() -> None:
#     """
#     データセットの動作確認用テスト関数です。
#     """
#     data_path = '../data/name_classfication.txt'
#     my_list_x, my_list_y = read_data(data_path)
#     my_dataset = NameClassDataset(my_list_x, my_list_y)
#
#     print(f"データセットの総件数: {len(my_dataset)}")
#     if len(my_dataset) > 0:
#         print(f"インデックス 0: {my_dataset[0]}")
#     if len(my_dataset) > 1:
#         print(f"インデックス 1: {my_dataset[1]}")
#     if len(my_dataset) > 2:
#         print(f"インデックス 2: {my_dataset[2]}")


if __name__ == '__main__':
    # 動作確認を行う場合はコメントアウトを解除してください
    # test_dataset()
    get_dataloader()
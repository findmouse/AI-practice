# coding:utf-8
import string
from typing import Tuple

import torch
from torch.utils.data import Dataset

# 利用可能な文字セット（アルファベット + 記号）
letters = string.ascii_letters + " ,.;'"
n_letters = len(letters)

# 分類対象の国籍リスト
categories = [
    'Italian', 'English', 'Arabic', 'Spanish', 'Scottish', 'Irish',
    'Chinese', 'Vietnamese', 'Japanese', 'French', 'Greek', 'Dutch',
    'Korean', 'Polish', 'Portuguese', 'Russian', 'Czech', 'German'
]


class NameClassDataset(Dataset):
    """
    名前（文字列）と国籍（ラベル）のデータセットを管理し、
    モデルに入力可能なテンソル形式に変換して返します。
    """

    def __init__(self, my_list_x: list, my_list_y: list) -> None:
        super().__init__()
        # 入力データ（名前のリスト）
        self.my_list_x = my_list_x
        # 正解ラベル（国籍のリスト）
        self.my_list_y = my_list_y
        # 総サンプル数
        self.sample_len = len(my_list_x)

    def __len__(self) -> int:
        """
        データセットの総サンプル数を返します。
        """
        return self.sample_len

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        指定されたインデックスのデータをOne-hotテンソル化して返します。
        """
        # インデックスの境界値チェックと補正 [0, self.sample_len - 1]
        index = min(max(index, 0), self.sample_len - 1)

        # データの抽出
        x = self.my_list_x[index]
        y = self.my_list_y[index]

        # 入力データ（名前）のOne-hotテンソル化
        tensor_x = torch.zeros(len(x), n_letters)
        for char_idx, letter in enumerate(x):
            letter_pos = letters.find(letter)
            if letter_pos != -1:
                tensor_x[char_idx][letter_pos] = 1.0

        # 正解ラベル（国籍）のインデックスをテンソル化
        tensor_y = torch.tensor(categories.index(y), dtype=torch.long)

        return tensor_x, tensor_y

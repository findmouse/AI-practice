# -*- coding: utf-8 -*-
from torch.utils.data import Dataset
import torch
import pickle
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class MyDataset(Dataset):
    """カスタム対話データセット: 切り詰めた input_ids テンソルを返す。"""

    def __init__(self, input_list, max_len):
        super().__init__()
        self.input_list = input_list
        self.max_len = max_len

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, index):
        input_ids = self.input_list[index][: self.max_len]
        return torch.tensor(input_ids, dtype=torch.long)


if __name__ == '__main__':
    pkl = os.path.join(ROOT, 'data', 'medical_valid.pkl')
    with open(pkl, 'rb') as f:
        input_list = pickle.load(f)
    dataset = MyDataset(input_list, max_len=300)
    print(len(dataset), dataset[0][:20])

# -*- coding: utf-8 -*-
import os
import sys
import pickle

import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from .dataset import MyDataset
except ImportError:
    from dataset import MyDataset

from parameter_config import ParameterConfig


def load_dataset(train_path, valid_path, max_len=300,
                 max_train_samples=None, max_valid_samples=None):
    with open(train_path, 'rb') as f:
        train_input_list = pickle.load(f)
    with open(valid_path, 'rb') as f:
        valid_input_list = pickle.load(f)

    if max_train_samples is not None:
        train_input_list = train_input_list[:max_train_samples]
    if max_valid_samples is not None:
        valid_input_list = valid_input_list[:max_valid_samples]

    train_dataset = MyDataset(train_input_list, max_len)
    val_dataset = MyDataset(valid_input_list, max_len)
    print(f'学習サンプル数: {len(train_dataset)}, 検証サンプル数: {len(val_dataset)}')
    return train_dataset, val_dataset


def collate_fn(batch):
    """
    1 batch の可変長系列を padding する:
    - input_ids は 0([PAD]) で埋める
    - labels は -100 で埋める（loss に含めない）
    """
    input_ids = rnn_utils.pad_sequence(batch, batch_first=True, padding_value=0)
    labels = rnn_utils.pad_sequence(batch, batch_first=True, padding_value=-100)
    return input_ids, labels


def get_dataloader(train_path, valid_path, batch_size=4, max_len=300,
                   max_train_samples=None, max_valid_samples=None):
    train_dataset, val_dataset = load_dataset(
        train_path, valid_path, max_len=max_len,
        max_train_samples=max_train_samples,
        max_valid_samples=max_valid_samples,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
    )
    validate_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
    )
    return train_dataloader, validate_dataloader


if __name__ == '__main__':
    params = ParameterConfig()
    train_loader, valid_loader = get_dataloader(
        params.train_path, params.valid_path, batch_size=params.batch_size
    )
    for input_ids, labels in valid_loader:
        print(input_ids.shape, labels.shape)
        print(input_ids[0][:20])
        print(labels[0][:20])
        break

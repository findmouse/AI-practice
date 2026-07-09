import torch
import torch.nn as nn


def init_zero():
    linear = nn.Linear(in_features=3, out_features=2, bias=True)
    nn.init.zeros_(linear.weight)
    nn.init.zeros_(linear.bias)
    print(linear.weight.data)
    print(linear.bias.data)


def init_one():
    linear = nn.Linear(in_features=3, out_features=2, bias=True)
    nn.init.ones_(linear.weight)
    nn.init.ones_(linear.bias)
    print(linear.weight.data)
    print(linear.bias.data)


def init_constant():
    linear = nn.Linear(in_features=3, out_features=2, bias=True)
    nn.init.constant_(linear.weight, 100)
    nn.init.constant_(linear.bias, 99)
    print(linear.weight.data)
    print(linear.bias.data)


def init_normal():
    linear = nn.Linear(in_features=3, out_features=2, bias=True)
    nn.init.normal_(linear.weight, mean=0, std=1)
    nn.init.normal_(linear.bias, mean=0, std=1)
    print(linear.weight.data)
    print(linear.bias.data)


def init_uniform():
    linear = nn.Linear(in_features=3, out_features=2, bias=True)
    # 0~1
    nn.init.uniform_(linear.weight)
    nn.init.uniform_(linear.bias)
    print(linear.weight.data)
    print(linear.bias.data)


def init_kaiming_normal():
    linear = nn.Linear(in_features=3, out_features=2, bias=True)
    nn.init.kaiming_normal_(linear.weight)
    nn.init.zeros_(linear.bias)
    print(linear.weight.data)
    print(linear.bias.data)


def init_kaiming_uniform():
    linear = nn.Linear(in_features=3, out_features=2, bias=True)
    nn.init.kaiming_uniform_(linear.weight)
    nn.init.zeros_(linear.bias)
    print(linear.weight.data)
    print(linear.bias.data)


def init_xavier_normal():
    linear = nn.Linear(in_features=3, out_features=2, bias=True)
    nn.init.xavier_normal_(linear.weight)
    nn.init.zeros_(linear.bias)
    print(linear.weight.data)
    print(linear.bias.data)


def init_xavier_uniform():
    linear = nn.Linear(in_features=3, out_features=2, bias=True)
    nn.init.xavier_uniform_(linear.weight)
    nn.init.zeros_(linear.bias)
    print(linear.weight.data)
    print(linear.bias.data)


if __name__ == '__main__':
    init_zero()
    init_one()
    init_constant()
    init_normal()
    init_uniform()
    init_kaiming_normal()
    init_kaiming_uniform()
    init_xavier_normal()
    init_xavier_uniform()

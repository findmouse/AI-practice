import torch.nn as nn
import torch


def test_CrossEntropyLoss1():
    y_true = torch.tensor([0, 1, 2], dtype=torch.int64)
    y_predict = torch.tensor([[18, 9, 10], [2, 14, 6], [3, 8, 16]], dtype=torch.float32)

    loss = nn.CrossEntropyLoss()
    print(loss(y_predict, y_true))


def test_CrossEntropyLoss2():
    y_true = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32)
    y_predict = torch.tensor([[18, 9, 10], [2, 14, 6], [3, 8, 16]], dtype=torch.float32)

    loss = nn.CrossEntropyLoss()
    print(loss(y_predict, y_true.argmax(dim=1)))


def test_BCELoss():
    y_true = torch.tensor([0, 1, 0, 1], dtype=torch.float32)
    # y_predict = torch.tensor([0.9, 0.1, 0.8, 0.2], dtype=torch.float32)
    y_predict = torch.tensor([0.1, 0.9, 0.2, 0.8], dtype=torch.float32)

    loss = nn.BCELoss()
    print(loss(y_predict, y_true))


def test_L1Loss():
    y_true = torch.tensor([2.0, 3.0, 1.0], dtype=torch.float32)
    y_predict = torch.tensor([1.0, 5.0, 4.0], dtype=torch.float32)

    loss = nn.L1Loss()
    print(loss(y_predict, y_true))


def test_MSELoss():
    y_true = torch.tensor([2.0, 3.0, 1.0], dtype=torch.float32)
    y_predict = torch.tensor([1.0, 5.0, 4.0], dtype=torch.float32)

    loss = nn.MSELoss()
    print(loss(y_predict, y_true))


def test_SmoothL1Loss():
    y_true = torch.tensor([2.0, 3.0, 1.0], dtype=torch.float32)
    y_predict = torch.tensor([1.0, 5.0, 4.0], dtype=torch.float32)

    loss = nn.SmoothL1Loss()
    print(loss(y_predict, y_true))


if __name__ == '__main__':
    # test_CrossEntropyLoss1()
    # test_CrossEntropyLoss2()
    # test_BCELoss()
    # test_L1Loss()
    # test_MSELoss()
    test_SmoothL1Loss()

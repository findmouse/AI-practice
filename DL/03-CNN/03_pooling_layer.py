import torch
import torch.nn as nn

# 单通道最大池化
def single_MaxPool2d():
    inputs = torch.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8]]]).float()
    print(inputs.shape)

    pooling = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
    output = pooling(inputs)
    print(output)

# 单通道平均池化
def single_AvgPool2d():
    inputs = torch.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8]]]).float()
    print(inputs.shape)

    pooling = nn.AvgPool2d(kernel_size=2, stride=1, padding=0)
    output = pooling(inputs)
    print(output)

# 多通道最大池化
def multiple_MaxPool2d():
    inputs = torch.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                           [[10, 20, 30], [40, 50, 60], [70, 80, 90]],
                           [[11, 22, 33], [44, 55, 66], [77, 88, 99]]]).float()
    print(inputs.shape)
    pooling = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
    output = pooling(inputs)
    print(output)

# 多通道平均池化
def multiple_AvgPool2d():
    inputs = torch.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                           [[10, 20, 30], [40, 50, 60], [70, 80, 90]],
                           [[11, 22, 33], [44, 55, 66], [77, 88, 99]]]).float()
    print(inputs.shape)
    pooling = nn.AvgPool2d(kernel_size=2, stride=1, padding=0)
    output = pooling(inputs)
    print(output)


if __name__ == '__main__':
    # single_MaxPool2d()
    # single_AvgPool2d()
    multiple_MaxPool2d()
    multiple_AvgPool2d()

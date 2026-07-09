import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def test_sigmoid():
    # 関数
    x = torch.linspace(-20, 20, 1000)
    y = torch.sigmoid(x)
    plt.plot(x, y)
    plt.grid(True)
    plt.show()

    # 导函数
    x = torch.linspace(-20, 20, 1000, requires_grad=True)
    torch.sigmoid(x).sum().backward()
    plt.plot(x.detach(), x.grad)
    plt.grid(True)
    plt.show()


def test_tanh():
    # 関数
    x = torch.linspace(-20, 20, 1000)
    y = torch.tanh(x)
    plt.plot(x, y)
    plt.grid(True)
    plt.show()

    # 导函数
    x = torch.linspace(-20, 20, 1000, requires_grad=True)
    torch.tanh(x).sum().backward()
    plt.plot(x.detach(), x.grad)
    plt.grid(True)
    plt.show()


def test_relu():
    # 関数
    x = torch.linspace(-20, 20, 1000)
    y = torch.relu(x)
    plt.plot(x, y)
    plt.grid(True)
    plt.show()

    # 导函数
    x = torch.linspace(-20, 20, 1000, requires_grad=True)
    torch.relu(x).sum().backward()
    plt.plot(x.detach(), x.grad)
    plt.grid(True)
    plt.show()


def test_softmax():
    x = torch.tensor([0.2, 0.02, 0.15, 0.15, 1.3, 0.5, 0.06, 1.1, 0.05, 3.75])
    res = torch.softmax(x, dim=0)
    print(res)
    print(x[torch.argmax(res).item()])

    max_val, max_idx = torch.max(res, dim=0)
    print(f"最大値: {max_val}, ポジション: {max_idx}")


if __name__ == '__main__':
    # test_sigmoid()
    # test_tanh()
    # test_relu()
    test_softmax()

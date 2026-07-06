import torch
import numpy as np


def test01():
    # データ（特徴量 + ターゲット）
    x = torch.tensor(5)
    y = torch.tensor(0.)

    # パラメータ（重み + バイアス）
    w = torch.tensor(1., requires_grad=True, dtype=torch.float32)
    b = torch.tensor(3., requires_grad=True, dtype=torch.float32)

    # 予測
    z = w * x + b

    # 損失
    # Mean Squared Error
    # 通常、回帰タスク（住宅価格、気温、株価などの連続値の予測など）に使用され、モデルの予測値と実際の値の間の「差」がどれだけあるかを測定します。
    loss = torch.nn.MSELoss()
    loss = loss(z, y)
    # 微分（偏導関数の計算）
    loss.backward()
    # 勾配
    print(f"w.grad-->{w.grad}")
    print(f"w.grad-->{b.grad}")


def test02():
    # データ（特徴量 + ターゲット）
    x = torch.ones(2, 5)
    y = torch.zeros(2, 3)

    # パラメータ（重み + バイアス）
    w = torch.randn(5, 3, requires_grad=True)
    b = torch.randn(3, requires_grad=True)

    # 予測
    z = torch.matmul(x, w) + b

    # 損失
    # Mean Squared Error
    # 通常、回帰タスク（住宅価格、気温、株価などの連続値の予測など）に使用され、モデルの予測値と実際の値の間の「差」がどれだけあるかを測定します。
    loss = torch.nn.MSELoss()
    loss = loss(z, y)
    # 微分（偏導関数の計算）
    loss.backward()
    # 勾配
    print(f"w.grad-->{w.grad}")
    print(f"w.grad-->{b.grad}")


if __name__ == '__main__':
    # test01()
    test02()

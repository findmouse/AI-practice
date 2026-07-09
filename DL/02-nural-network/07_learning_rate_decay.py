import torch
import matplotlib.pyplot as plt


def test_StepLR():
    # パラメータの初期化
    lr0 = 0.1
    iter = 100
    epoches = 200

    # ネットワークの初期化 (※元の「网格」はネットワーク/モデルの文脈として翻訳)
    x = torch.tensor([1.0])
    w = torch.tensor([1.0], requires_grad=True)
    y = torch.tensor([1.0])

    # オプティマイザ
    optimizer = torch.optim.SGD([w], lr=lr0, momentum=0.9)

    # 学習率のスケジューリング
    scheduluer_lr = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # エポックのループ
    epoch_ist = []
    lr_list = []

    for epoch in range(epoches):
        lr_list.append(scheduluer_lr.get_last_lr())
        epoch_ist.append(epoch)

        # ミニバッチのループ
        for i in range(iter):
            # 損失の計算
            loss = (w * x - y) ** 2 * 0.5
            # パラメータの更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 学習率の更新
        scheduluer_lr.step()

    # グラフの描画
    plt.plot(epoch_ist, lr_list)
    plt.grid(True)
    plt.show()


def test_MultiStepLR():
    # パラメータの初期化
    lr0 = 0.1
    iter = 100
    epoches = 200

    # ネットワークの初期化
    x = torch.tensor([1.0])
    w = torch.tensor([1.0], requires_grad=True)
    y = torch.tensor([1.0])

    # オプティマイザ
    optimizer = torch.optim.SGD([w], lr=lr0, momentum=0.9)

    # 学習率のスケジューリング
    scheduluer_lr = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 70, 100, 120, 140, 160, 180],
                                                         gamma=0.5)

    # エポックのループ
    epoch_ist = []
    lr_list = []

    for epoch in range(epoches):
        lr_list.append(scheduluer_lr.get_last_lr())
        epoch_ist.append(epoch)

        # ミニバッチのループ
        for i in range(iter):
            # 損失の計算
            loss = (w * x - y) ** 2 * 0.5
            # パラメータの更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 学習率の更新
        scheduluer_lr.step()

    # グラフの描画
    plt.plot(epoch_ist, lr_list)
    plt.grid(True)
    plt.show()


# 指数関数的な学習率の減衰
def test_ExponentialLR():
    # パラメータの初期化
    lr0 = 0.1
    iter = 100
    epoches = 200

    # ネットワークの初期化
    x = torch.tensor([1.0])
    w = torch.tensor([1.0], requires_grad=True)
    y = torch.tensor([1.0])

    # オプティマイザ
    optimizer = torch.optim.SGD([w], lr=lr0, momentum=0.9)

    # 学習率のスケジューリング
    scheduluer_lr = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    # エポックのループ
    epoch_ist = []
    lr_list = []

    for epoch in range(epoches):
        lr_list.append(scheduluer_lr.get_last_lr())
        epoch_ist.append(epoch)

        # ミニバッチのループ
        for i in range(iter):
            # 損失の計算
            loss = (w * x - y) ** 2 * 0.5
            # パラメータの更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 学習率の更新
        scheduluer_lr.step()

    # グラフの描画
    plt.plot(epoch_ist, lr_list)
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    # test_StepLR()
    # test_MultiStepLR()
    test_ExponentialLR()
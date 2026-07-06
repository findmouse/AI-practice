import torch
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def create_dataset() -> tuple:
    """線形回帰用のデータセットを生成する。"""
    return make_regression(
        n_samples=100,
        n_features=1,
        noise=10,
        bias=1.5,
        coef=True
    )


def create_dataloader(x_data, y_data, batch_size=2) -> DataLoader:
    """データローダーを生成する。"""
    dataset = TensorDataset(
        torch.FloatTensor(x_data),
        torch.FloatTensor(y_data)
    )

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False
    )


def train_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    epochs: int
) -> list:
    """モデルを学習する。"""

    loss_history = []

    model.train()

    for _ in range(epochs):

        total_loss = 0.0
        sample_count = 0

        for x_batch, y_batch in dataloader:

            # 勾配を初期化
            optimizer.zero_grad()

            # 順伝播
            pred_y = model(x_batch.float())

            # 損失を計算
            loss = criterion(
                pred_y,
                y_batch.reshape(-1, 1).float()
            )

            # 誤差逆伝播
            loss.backward()

            # パラメータ更新
            optimizer.step()

            total_loss += loss.item()
            sample_count += x_batch.size(0)

        loss_history.append(total_loss / sample_count)

    return loss_history


def plot_loss(loss_history: list) -> None:
    """損失の推移を表示する。"""

    plt.figure(figsize=(8, 4))
    plt.plot(range(len(loss_history)), loss_history)
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()


def plot_regression_result(
    model: nn.Module,
    x_data,
    coef: float,
    bias: float = 1.5
) -> None:
    """学習結果と正解直線を表示する。"""

    model.eval()

    x_line = torch.linspace(
        float(x_data.min()),
        float(x_data.max()),
        1000
    )

    with torch.no_grad():
        pred_y = model(x_line.reshape(-1, 1))

    true_y = x_line * coef + bias

    plt.figure(figsize=(8, 4))
    plt.plot(
        x_line.numpy(),
        pred_y.numpy(),
        label="学習結果"
    )
    plt.plot(
        x_line.numpy(),
        true_y.numpy(),
        label="正解"
    )

    plt.title("Linear Regression")
    plt.grid(True)
    plt.legend()
    plt.show()


def main() -> None:
    """メイン処理"""

    # データセット生成
    x_data, y_data, coef = create_dataset()

    # データローダー生成
    dataloader = create_dataloader(
        x_data=x_data,
        y_data=y_data,
        batch_size=2
    )

    # モデル生成
    model = nn.Linear(1, 1)

    # 損失関数
    criterion = nn.MSELoss()

    # オプティマイザ
    optimizer = optim.SGD(
        model.parameters(),
        lr=1e-3
    )

    # 学習
    loss_history = train_model(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=100
    )

    # 損失推移を表示
    plot_loss(loss_history)

    # 回帰結果を表示
    plot_regression_result(
        model=model,
        x_data=x_data,
        coef=coef
    )


if __name__ == "__main__":
    main()
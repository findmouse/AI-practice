import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from tqdm import tqdm


# データセットの作成
def create_dataset():
    # CSVファイルを読み込む
    data = pd.read_csv('data/phone_price.csv')

    # 特徴量(X)とラベル(y)に分割
    x, y = data.iloc[:, :-1], data.iloc[:, -1]

    # PyTorchで扱いやすいデータ型へ変換
    x = x.astype(np.float32)
    y = y.astype(np.int64)

    # 学習データと検証データへ分割
    # stratify=y により各クラスの割合を維持したまま分割する
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=88,
        stratify=y,
        shuffle=True
    )

    # 標準化に使用する平均値と標準偏差を学習データから計算
    mean = x_train.mean()
    std = x_train.std()

    # 標準偏差が0の特徴量によるゼロ除算を防ぐ
    std = std.replace(0, 1)

    # 学習データの統計量を利用して標準化
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    # TensorDatasetへ変換
    train_dataset = TensorDataset(
        torch.from_numpy(x_train.values.astype(np.float32)),
        torch.tensor(y_train.values, dtype=torch.int64)
    )

    val_dataset = TensorDataset(
        torch.from_numpy(x_test.values.astype(np.float32)),
        torch.tensor(y_test.values, dtype=torch.int64)
    )

    # DataLoaderを作成
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 入力特徴量数とクラス数を返す
    return train_loader, val_loader, x_train.shape[1], len(np.unique(y))


# ニューラルネットワークモデルの定義
class PhonePriceModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PhonePriceModel, self).__init__()

        # 全結合層
        self.linear1 = nn.Linear(input_dim, 64)
        self.linear2 = nn.Linear(64, 128)
        self.linear3 = nn.Linear(128, output_dim)

        # 過学習を抑えるためのDropout
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # Linear → Dropout → ReLU
        x = torch.relu(self.dropout(self.linear1(x)))
        x = torch.relu(self.dropout(self.linear2(x)))

        # 出力層
        # CrossEntropyLossを使用するためSoftmaxは不要
        output = self.linear3(x)

        return output


def train():
    # GPUが利用可能ならGPUを使用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # データセットを読み込む
    train_loader, valid_loader, input_dim, class_num = create_dataset()

    # モデルを作成してGPU/CPUへ転送
    phone_model = PhonePriceModel(input_dim, class_num).to(device)

    # 多クラス分類用の損失関数
    criterion = nn.CrossEntropyLoss()

    # Adamオプティマイザ
    # weight_decayはL2正則化として機能する
    optimizer = optim.Adam(
        phone_model.parameters(),
        lr=1e-3,
        weight_decay=1e-4
    )

    epochs = 100

    # 最良の検証損失を保存するため、初期値を無限大に設定
    best_val_loss = float("inf")

    # 最良モデルの検証精度
    best_val_acc = 0

    # Early Stoppingの設定
    patience = 10
    patience_count = 0

    # モデル保存先
    save_path = 'data/predict_phone_price.pth'

    for epoch in range(epochs):

        # ==========================
        # 学習フェーズ
        # ==========================
        phone_model.train()

        losses_sum = 0
        sample_count = 0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            x = x.to(device)
            y = y.to(device)

            # 順伝播
            y_predict = phone_model(x)

            # 損失計算
            loss = criterion(y_predict, y)

            # 勾配を初期化
            optimizer.zero_grad()

            # 誤差逆伝播
            loss.backward()

            # パラメータ更新
            optimizer.step()

            batch_size = x.shape[0]

            # 各バッチの損失をサンプル数で重み付けして加算
            losses_sum += loss.item() * batch_size
            sample_count += batch_size

        # ==========================
        # 検証フェーズ
        # ==========================
        phone_model.eval()

        val_loss_sum = 0
        correct = 0
        total = 0

        # 検証時は勾配計算を無効化して高速化・省メモリ化
        with torch.no_grad():

            for x, y in valid_loader:
                x = x.to(device)
                y = y.to(device)

                y_predict = phone_model(x)

                loss = criterion(y_predict, y)

                # 最大確率のクラスを予測結果とする
                pred = torch.argmax(y_predict, dim=1)

                batch_size = x.shape[0]

                val_loss_sum += loss.item() * batch_size

                # 正解数をカウント
                correct += (pred == y).sum().item()
                total += batch_size

        # エポック全体の平均損失と精度を計算
        train_loss = losses_sum / sample_count
        val_loss = val_loss_sum / total
        val_acc = correct / total

        # ==========================
        # 最良モデルの保存
        # ==========================
        if val_loss < best_val_loss:

            best_val_loss = val_loss
            best_val_acc = val_acc

            # Early Stoppingカウンタをリセット
            patience_count = 0

            # 最良モデルの重みを保存
            torch.save(phone_model.state_dict(), save_path)

            best_message = "saved best model"

        else:

            patience_count += 1
            best_message = f"early stop count: {patience_count}/{patience}"

        print(
            f"epoch: {epoch + 1}, "
            f"train_loss: {train_loss:.4f}, "
            f"val_loss: {val_loss:.4f}, "
            f"val_acc: {val_acc:.4f}, "
            f"{best_message}"
        )

        # 一定回数改善がなければ学習終了
        if patience_count >= patience:
            print(f"early stopping at epoch {epoch + 1}")
            break

    print(
        f"best_val_loss: {best_val_loss:.4f}, "
        f"best_val_acc: {best_val_acc:.4f}"
    )


if __name__ == '__main__':
    train()

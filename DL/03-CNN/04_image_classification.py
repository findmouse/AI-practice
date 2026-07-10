# PyTorchのコアライブラリ。テンソル計算、モデル学習、GPU/CPUデバイス管理に使用される
import torch
# PyTorchのニューラルネットワークモジュール。畳み込み層、プーリング層、損失関数などのネットワークコンポーネントを提供
import torch.nn as nn
from torchsummary import summary
# torchvisionに組み込まれているCIFAR10データセットの読み込みツール
from torchvision.datasets import CIFAR10
# 画像データをPyTorchのTensorに変換し、ニューラルネットワークへの入力を容易にする
from torchvision.transforms import ToTensor
# 複数の画像前処理操作を順番に組み合わせる
from torchvision.transforms import Compose
# PyTorchの最適化モジュール。SGDやAdamなどのモデルパラメータ更新に使用される
import torch.optim as optim
# データローダー。データをバッチごとに読み込み、シャッフルやマルチプロセス読み込みをサポート
from torch.utils.data import DataLoader
# tqdm進捗バー。学習の進捗状況を表示するために使用される
from tqdm import tqdm

EPOCHS = 10
BATCH_SIZE = 512
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_train_dataloader():
    # データの取得
    train_data = CIFAR10(root='./data', train=True, transform=Compose([ToTensor()]))
    return DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)


def get_valid_dataloader():
    # データの取得
    valid_data = CIFAR10(root='./data', train=False, transform=Compose([ToTensor()]))
    return DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False)


class ImgClassification(nn.Module):
    def __init__(self):
        super(ImgClassification, self).__init__()
        self.layer1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pooling1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pooling2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer3 = nn.Linear(in_features=2048, out_features=120)
        self.layer4 = nn.Linear(in_features=120, out_features=84)
        self.out = nn.Linear(in_features=84, out_features=10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.pooling1(x)

        x = self.relu(self.layer2(x))
        x = self.pooling2(x)

        x = x.view(x.size(0), -1)
        x = self.relu(self.layer3(x))
        x = self.relu(self.layer4(x))

        out = self.out(x)
        return out


def train(model, train_dataloader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_mean = []
    for epoch in range(EPOCHS):

        model.train()
        loss_sum = 0
        sample = 0
        pbar = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{EPOCHS}')
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)
            y_pre = model(x)
            loss = criterion(y_pre, y)
            loss_sum += loss.item()
            sample += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=f'{loss_sum / sample:.4f}')
        loss_mean.append(loss_sum / sample)
        print(loss_sum / sample)
    print(loss_mean)
    torch.save(model.state_dict(), './data/model.pth')


def test(model, test_dataloader):
    model.load_state_dict(torch.load('./data/model.pth', map_location=device))
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in tqdm(test_dataloader, desc='Testing'):
            x = x.to(device)
            y = y.to(device)
            y_pre = model(x)
            pred = y_pre.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    acc = correct / total
    print(f'Test Accuracy: {acc:.4f} ({correct}/{total})')
    return acc


if __name__ == '__main__':
    model = ImgClassification().to(device)
    # summary(model, input_size=(3, 32, 32))
    # train(model, get_train_dataloader())
    test(model, get_valid_dataloader())
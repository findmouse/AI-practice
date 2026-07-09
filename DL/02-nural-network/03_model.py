import torch
import torch.nn as nn
from torchsummary import summary


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(3, 3)
        nn.init.xavier_uniform_(self.linear1.weight)

        self.linear2 = nn.Linear(3, 2)
        nn.init.kaiming_normal_(self.linear2.weight)

        self.out = nn.Linear(2, 2)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.sigmoid(x)

        x = self.linear2(x)
        x = torch.relu(x)

        x = self.out(x)
        x = torch.softmax(x, dim=-1)

        return x


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device-->{device}")

    my_model = Model().to(device)
    my_data = torch.randn(5, 3).to(device)
    print(f"my_data.shape-->{my_data.shape}")
    out = my_model(my_data)
    print(f"out.shape-->{out.shape}")
    summary(my_model, input_size=(3,), batch_size=5)

    print("*" * 40 + "w&b" + "*" * 40)
    for name, param in my_model.named_parameters():
        print(name, param)

import torch
import numpy as np

data_np = np.random.rand(2, 3)
print(data_np)
print(torch.tensor(data_np))

data = torch.arange(0, 10, 2)
print(f"data-->{data}")

data = torch.linspace(0, 9, 5)
print(f"data-->{data}")

data = torch.randn(2, 3)
print(f"data-->{data}")

seed = torch.random.initial_seed()
print("随机数种子：", seed)

torch.random.manual_seed(100)
data = torch.randn(2, 3)
print(data)
print("随机数种子：", torch.random.initial_seed())

import torch
import numpy as np

torch.random.manual_seed(22)
data = torch.randint(0, 10, [3, 4], dtype=torch.float32)

print("-" * 40 + "data" + "-" * 40)
print(f"data-->{data}")

print("-" * 40 + "mean" + "-" * 40)
print(f"data.mean()-->{data.mean()}")
print(f"data.mean(dim=0)-->{data.mean(dim=0)}")
print(f"data.mean(dim=1)-->{data.mean(dim=1)}")

print("-" * 40 + "sum" + "-" * 40)
print(f"data.sum()-->{data.sum()}")
print(f"data.sum(dim=0)-->{data.sum(dim=0)}")
print(f"data.sum(dim=1)-->{data.sum(dim=1)}")

print("-" * 40 + "sqrt" + "-" * 40)
print(f"data.sqrt()-->{data.sqrt()}")

print("-" * 40 + "pow" + "-" * 40)
print(f"torch.pow(data, 2)-->{torch.pow(data, 2)}")

print("-" * 40 + "pow" + "-" * 40)
print(f"torch.pow(2, data)-->{torch.pow(2, data)}")

print("-" * 40 + "exp" + "-" * 40)
print(f"data.exp()-->{data.exp()}")

print("-" * 40 + "log" + "-" * 40)
print(f"data.log()-->{data.log()}")

print("-" * 40 + "log2" + "-" * 40)
print(f"data.log2()-->{data.log2()}")

print("-" * 40 + "log10" + "-" * 40)
print(f"data.log10()-->{data.log10()}")

import torch
import numpy as np

torch.random.manual_seed(22)
data1 = torch.randint(0, 20, [2, 3])
print(f"data1-->{data1}")

torch.random.manual_seed(23)
data2 = torch.randint(0, 20, [3, 4])
print(f"data2-->{data2}")

print("-" * 40 + "matmul" + "-" * 40)
print(data1.matmul(data2))

print("-" * 40 + "@" + "-" * 40)
print(data1 @ data2)

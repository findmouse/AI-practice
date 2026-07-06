import torch
import numpy as np

torch.random.manual_seed(22)
data1 = torch.randint(0, 20, [2, 3])
print(f"data1-->{data1}")

torch.random.manual_seed(23)
data2 = torch.randint(0, 20, [2, 3])
print(f"data2-->{data2}")
print("-" * 40 + "mul 10" + "-" * 40)
print(torch.mul(data1, data2))
print("-" * 40 + "* 10" + "-" * 40)
print(data1 * data2)

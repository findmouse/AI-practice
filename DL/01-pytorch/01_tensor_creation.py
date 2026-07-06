import torch
import numpy as np

print(10)
print(torch.tensor(10))
print(torch.tensor([10, ]))

data = torch.zeros(2, 3)
print(f"data-->{data}")

data1 = torch.zeros_like(data)
print(f"data1-->{data1}")

data = torch.ones(2, 3)
print(f"data-->{data}")

data1 = torch.ones_like(data)
print(f"data1-->{data1}")

data = torch.full([2, 3], 11)
print(f"data-->{data}")

data1 = torch.full_like(data, 22)
print(f"data1-->{data1}")

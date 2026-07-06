import torch
import numpy as np

data = [[1., 2., 3.], [4., 5., 6.]]
print(data)
print(torch.tensor(data))

data = torch.Tensor(2, 3)
print(f"data-->{data}")
print(f"data-->{data.dtype}")
print(torch.Tensor([2, 3]))

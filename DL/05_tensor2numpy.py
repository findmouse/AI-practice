import torch
import numpy as np

data_tensor = torch.tensor([2, 3, 4])
print(f"data_tensor.dtype-->{type(data_tensor)}")

data_numpy = data_tensor.numpy()
print(f"data_numpy.dtype-->{type(data_numpy)}")
# data_tensor[0]=10
data_tensor[0] = 100
print(f"data_tensor-->{data_tensor}")
print(f"data_numpy-->{data_numpy}")


new_numpy = data_tensor.numpy().copy()
new_numpy[0] = 200
print(f"new_numpy-->{new_numpy}")
print(f"data_tensor-->{data_tensor}")
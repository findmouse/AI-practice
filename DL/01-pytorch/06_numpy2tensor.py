import torch
import numpy as np

data_numpy = np.array([2, 3, 4])
data_tensor = torch.from_numpy(data_numpy)
print(f"data_numpy-->{data_numpy}")
print(f"type(data_numpy)-->{type(data_numpy)}")
print(f"data_tensor-->{data_tensor}")
print(f"type(data_tensor)-->{type(data_tensor)}")
print("*" * 80)
data_numpy[0] = 300
print(f"data_numpy-->{data_numpy}")
print(f"data_tensor-->{data_tensor}")

data_tensor2 = torch.tensor(data_numpy)
print("*" * 80)
data_numpy[0] = 400
print(f"data_numpy-->{data_numpy}")
print(f"data_tensor2-->{data_tensor2}")

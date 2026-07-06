import torch
import numpy as np

# torch.random.manual_seed(22)
# data = torch.randint(0, 10, [3, 4, 5])
# print(f"data-->{data}")
#
# print("-" * 40 + "data.shape" + "-" * 40)
# print(f"data.shape-->{data.shape}")
# print(f"data.shape-->{data.shape[0]}")
# print(f"data.shape-->{data.shape[1]}")
# print(f"data.shape-->{data.shape[2]}")
#
# print("-" * 40 + "data.size()" + "-" * 40)
# print(f"data.size()-->{data.size()}")
# print(f"data.size(0)-->{data.size(0)}")
# print(f"data.size(1)-->{data.size(1)}")
# print(f"data.size(2)-->{data.size(2)}")
#
# data = data.reshape(4, 5, 3)
# print("-" * 40 + "data.reshape" + "-" * 40)
# print(f"data.shape-->{data.shape}")
#
# data = data.reshape(2, -1, 3)
# print("-" * 40 + "data.reshape" + "-" * 40)
# print(f"data.shape-->{data.shape}")
#
# #
# # data = data.reshape(-1)
# # print("-" * 40 + "data.reshape(-1)" + "-" * 40)
# # print(f"data.shape-->{data.shape}")
#
#
# print("-" * 40 + "data.unsqueeze(dim=1)" + "-" * 40)
# data1 = data.unsqueeze(dim=1).unsqueeze(dim=-1)
# print(f"data.shape-->{data.shape}")
#
# print("-" * 40 + "data.unsqueeze(dim=1)" + "-" * 40)
# data1 = data.unsqueeze(dim=1).unsqueeze(-1)
# print(f"data.shape-->{data1.shape}")
#
# print("-" * 40 + "unsqueeze" + "-" * 40)
# print(data1.squeeze().shape)
#
# print("*" * 80)
# torch.random.manual_seed(23)
# data = torch.randint(0, 10, [4, 2, 3, 5])
# print(data.shape)
# print("-" * 40 + "[3,4,5,2]" + "-" * 40)
# # data.shape-->[3,5,4,2]
# data = torch.transpose(data, 0, 2)
# data = torch.transpose(data, 1, 3)
# print(data.shape)
#
# print("-" * 40 + "[4,2,3,5]" + "-" * 40)
# data = torch.permute(data, [2, 3, 0, 1])
# print(data.shape)
#
# print("*" * 80)
# torch.random.manual_seed(22)
# data = torch.randint(0, 10, [2, 3])
# print(f"data.shape-->{data.shape}")
# print(f"data.is_contiguous()-->{data.is_contiguous()}")
# print(data.view(-1).shape)
# data = torch.transpose(data, 0 ,1)
# print(f"data.shape-->{data.shape}")
# print(f"data.is_contiguous()-->{data.is_contiguous()}")
#
# data = data.contiguous()
# data = data.view(-1)
# print(f"data.shape-->{data.shape}")
# print(f"data.is_contiguous()-->{data.is_contiguous()}")


torch.random.manual_seed(22)
data1 = torch.randint(0, 10, [4, 3, 5])
print(f"data1.shape-->{data1.shape}")

torch.random.manual_seed(23)
data2 = torch.randint(0, 10, [4, 3, 5])
print(f"data2.shape-->{data2.shape}")

print("-" * 40 + "dim=0" + "-" * 40)
print(torch.cat([data2, data1], dim=0).shape)

print("-" * 40 + "dim=1" + "-" * 40)
print(torch.cat([data2, data1], dim=1).shape)

print("-" * 40 + "dim=2" + "-" * 40)
print(torch.cat([data2, data1], dim=2).shape)

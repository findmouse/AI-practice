import torch
import numpy as np

# torch.random.manual_seed(22)
# data = torch.randint(0, 10, [4, 5])
# print(f"data-->{data}")
#
# print("-" * 40 + "data[0]" + "-" * 40)
# print(f"data[0]-->{data[0]}")
#
# print("-" * 40 + "data[:,0]" + "-" * 40)
# print(f"data[:,0]-->{data[:, 0]}")
#
# print("-" * 40 + "data[2,3]" + "-" * 40)
# print(f"data[2,3]-->{data[2, 3]}")
#
# print("-" * 40 + "data[[1,2], [2,4]]" + "-" * 40)
# print(f"data[[1,2], [2,4]]-->{data[[1, 2], [2, 4]]}")
#
# print("-" * 40 + "data[[[1], [2]], [2, 4]]" + "-" * 40)
# print(f"data[[[1], [2]], [2, 4]]-->{data[[[1], [2]], [2, 4]]}")
#
# print("-" * 40 + "np.ix_" + "-" * 40)
# print(f"data[[[1], [2]], [2, 4]]-->{data[np.ix_([1, 2], [2, 4])]}")
#
# print("-" * 40 + "data[:2, 1:4]" + "-" * 40)
# print(f"data[:2, 1:4]-->{data[:2, 1:4]}")
#
# print("-" * 40 + "data[:2, 1:4:2]" + "-" * 40)
# print(f"data[:2, 1:4:2]-->{data[:2, 1:4:2]}")
#
# index = data[:, 2] > 2
# print(f"index-->{index}")
# print(data[index])
#
# print("-" * 80)
# index = data[2] > 2
# print(f"index-->{index}")
# print(data[:, index])

print("-" * 80)
torch.random.manual_seed(22)
data = torch.randint(0, 10, [3, 4, 5])
print(f"data-->{data}")

print("-" * 40 + "data[1]" + "-" * 40)
print(f"data[1]-->{data[1]}")

print("-" * 40 + "data[:,:,3]" + "-" * 40)
print(f"data[:,:,3]-->{data[:, :, 3]}")

print("-" * 40 + "data[1,2,2]" + "-" * 40)
print(f"data[1,2,2]-->{data[1, 2, 2]}")

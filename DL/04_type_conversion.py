import torch
import numpy as np

print(torch.IntTensor([2.3, 4.9]))
print(torch.LongTensor([2.3, 4.9]))
# FloatTensor-->float32， DoubleTensor-->float64
print(torch.FloatTensor([2.3, 4.9]))
data = torch.DoubleTensor([2.3, 4.9])
print(f"data.DoubleTensor-->{data}")
print(f"data.DoubleTensor.dtype-->{data.dtype}")
print(data)

print(f"data.dtype-->{data.dtype}")

data = data.type(torch.DoubleTensor)

print(f"类型变换后的data.dtype-->{data.dtype}")
# 转换为其他类型
data = data.type(torch.ShortTensor)
print(f"data.dtype-->{data.dtype}")
data = data.type(torch.IntTensor)
print(f"data.dtype-->{data.dtype}")
data = data.type(torch.LongTensor)
print(f"data.dtype-->{data.dtype}")
data = data.type(torch.FloatTensor)
print(f"data.dtype-->{data.dtype}")

data = torch.full([2, 3], 33)
print(f"data.dtype-->{data.dtype}")

data = data.short()
print(f"data.dtype-->{data.dtype}")
data = data.int()
print(f"data.dtype-->{data.dtype}")
data = data.long()
print(f"data.dtype-->{data.dtype}")
data = data.float()
print(f"data.dtype-->{data.dtype}")
data = data.double()
print(f"data.dtype-->{data.dtype}")

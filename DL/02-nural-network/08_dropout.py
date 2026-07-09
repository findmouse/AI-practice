import torch
import torch.nn as nn



layer = nn.Linear(in_features=4, out_features=5)
input = torch.randn([1,4])
y = layer(input)
print(y)

dropout = nn.Dropout(p=0.60)
y = dropout(y)
print(y)
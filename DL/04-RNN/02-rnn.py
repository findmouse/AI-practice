import torch
import torch.nn as nn


rnn = nn.RNN(input_size=128,hidden_size=64,num_layers=10)
x = torch.randn(12,24,128)
h = torch.zeros(10,24,64)
y,hn=rnn(x,h)
print(y.shape)
print(h.shape)
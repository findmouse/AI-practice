import torch
import torch.nn as nn
import matplotlib.pyplot as plt

img = plt.imread("./data/img.jpg")
print(img.shape)
img = torch.tensor(img).permute([2, 0, 1])
img = img.to(torch.float32).unsqueeze(0)
print(img.shape)

layer = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=(3,5), stride=(2,3), padding=1)

fm = layer(img)
print(fm.shape)

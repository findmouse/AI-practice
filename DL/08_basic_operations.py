import torch
import numpy as np

torch.random.manual_seed(22)
data = torch.randint(0, 10, [2, 3])

print(data)
print("-"*40+"add 10"+"-"*40)
print(data.add(10))
print(data)
print(data.add_(50))
print(data)

print("-"*40+"sub 10"+"-"*40)
print(data.sub(10))

print("-"*40+"mul 10"+"-"*40)
print(data.mul(10))


print("-"*40+"div 10"+"-"*40)
print(data.div(10))

print("-"*40+"neg 10"+"-"*40)
print(data.neg())
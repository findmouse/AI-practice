import torch
import matplotlib.pyplot as plt

torch.manual_seed(11)
temperature = torch.randint(0, 40, [30])

days = torch.range(1, 30)
plt.plot(days, temperature, label="temperature")
plt.scatter(days, temperature, label="temperature points")

t_avg = []
beta = 0.5
for i, temp in enumerate(temperature):
    if i == 0:
        t_avg.append(temp)
        continue
    t2 = beta * t_avg[i - 1] + (1 - beta) * temp
    t_avg.append(t2)

plt.plot(days, t_avg, label="weighted avg beta=0.5")

t_avg = []
beta = 0.9
for i, temp in enumerate(temperature):
    if i == 0:
        t_avg.append(temp)
        continue
    t2 = beta * t_avg[i - 1] + (1 - beta) * temp
    t_avg.append(t2)

plt.plot(days, t_avg, label="weighted avg beta=0.9")

plt.legend()
plt.show()

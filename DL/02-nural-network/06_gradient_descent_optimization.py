import torch
from sympy.abc import alpha


def testSGD():
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)

    loss = ((w ** 2) * 0.5).sum()
    optimizer = torch.optim.SGD([w], lr=0.1)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(w.grad)
    print(w.detach())
    print("*" * 80)

    loss = ((w ** 2) * 0.5).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(w.grad)
    print(w.detach())
    print("*" * 80)

    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)

    loss = ((w ** 2) * 0.5).sum()
    optimizer = torch.optim.SGD([w], lr=0.1, momentum=0.9)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(w.grad)

    print(w.detach())
    print("*" * 80)

    loss = ((w ** 2) * 0.5).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(w.grad)
    print(w.detach())
    print("*" * 80)


def test_adagrad():
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)

    loss = ((w ** 2) * 0.5).sum()
    optimizer = torch.optim.Adagrad([w], lr=0.1)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(w.grad)
    print(w.detach())
    print("*" * 80)

    loss = ((w ** 2) * 0.5).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(w.grad)
    print(w.detach())
    print("*" * 80)


def test_RMSprop():
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)

    loss = ((w ** 2) * 0.5).sum()
    optimizer = torch.optim.RMSprop([w], lr=0.1, alpha=0.9)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(w.grad)
    print(w.detach())
    print("*" * 80)

    loss = ((w ** 2) * 0.5).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(w.grad)
    print(w.detach())
    print("*" * 80)


if __name__ == '__main__':
    # testSGD()
    # test_adagrad()
    test_RMSprop()

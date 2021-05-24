import torch
import torch.nn as nn
import torch.nn.functional as F


# f(x) = x**3 - 2x
# f'(x) = 3x**2 - 2

x = torch.tensor(2.0, requires_grad=True)
fx = x**3 - 2*x
# print(fx)
# print(x.grad) # each tensor has two parts-- a value "bucket" and a grad "bucket"

fx.backward()

# print(x.grad)


# f(x,y) = 2*x*y - y**3
# partial_x(f)(x,y) = 2*y, partial_y(f)(x,y) = 2*x - 3*y**2
xy = torch.tensor([1.0, -1.0], requires_grad=True)
fxy = 2*xy[0]*xy[1] - xy[1]**3
print(fxy)
print(xy.grad)

fxy.backward()

print(xy.grad)


# caveats: to apply `.backward()` you have to apply it to a one element tensor
x = torch.rand(3, 2, requires_grad=True)
print(x)

(x**2).backward() # this computes reverse-mode autodiff, which really only applies to functions of many variables -> a single number

# when is backward used? in neural nets, you plug in a giant tensor of inputs, and the output is a single loss scalar


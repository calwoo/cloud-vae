# lets build a basic neural net

# lets try and solve the xor problem
# 0 1 -> xor(0,1) = 1
# 0 0 -> xor(0,0) = 0
# 1 1 -> xor(1,1) = 0

# inputs will be bit pairs e.g (0,1), outputs will be a single bit 0 or 1
import torch
import torch.nn as nn


# data
# this is a supervised learning problem, so we need to have an input x, output y
x = torch.tensor([[0,0], [0,1], [1,0], [1,1]], dtype=torch.float32)
y = torch.tensor([0,1,1,0], dtype=torch.long)

# print(x)
# print(y)

# model
# model = nn.Linear(2, 1)  # is linear regression

# linear regression doesn't cut the butter
model = nn.Sequential(nn.Linear(2, 3),
                      nn.Sigmoid(),
                      nn.Linear(3, 1))

# a linear layer under the hood is
# class OurLinear:
#     def __init__(self, inp_dim, out_dim):
#         self.W = torch.rand(inp_dim, out_dim)
#         self.b = torch.ones(out_dim)

#     def forward(self, x):
#         return self.W @ x + self.b

# optimizer
# optimizer = torch.optim.SGD()

# train loop
lr = 0.001
epochs = 1000
# each epoch, we'll pass the data x through the model, capture the output, compare to loss, backprop the loss, upgrade gradients, repeat
for epoch in range(epochs):
    # pass x through model
    out = model.forward(x)
    
    # compute loss
    loss = torch.mean((out - y)**2)
    # backprop loss
    loss.backward()
    # upgrade gradients (whenever I write w.grad, mathematically you say "dloss/dw") (w <- w - eta * dL/dw)
    # gradient descent takes weights w ~> w - learning_rate * w.grad

    # the first GOTCHA with loss.backward() is that it doesn't automatically zero the gradients
    for w in model.parameters():
        w.data = w.data - lr * w.grad.data
        w.grad.data = torch.zeros_like(w.grad)
    
    print(f"epoch {epoch} has loss {loss.item()}")

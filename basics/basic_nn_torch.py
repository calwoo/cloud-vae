import numpy as np

import torch
import torch.nn as nn


# data
x = torch.tensor([[0,0], [0,1], [1,0], [1,1]], dtype=torch.float32)
y = torch.tensor([0,1,1,0], dtype=torch.float32).view(-1, 1)

# model
model = nn.Sequential(nn.Linear(2, 3),
                      nn.Sigmoid(),
                      nn.Linear(3, 1))

# optimizer and loss
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
criterion = nn.MSELoss()

# train loop
epochs = 3000
batch_size = 4
for epoch in range(epochs):
    avg_loss = 0
    for i in range(batch_size):
        # randomly sample a datapoint from x
        random_idx = np.random.choice(x.size(0))
        x_pt = x[random_idx]
        y_pt = y[random_idx]

        out = model.forward(x_pt)
        loss = criterion(out, y_pt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss += loss.item()
    avg_loss = avg_loss / batch_size

    print(f"epoch {epoch} has loss {avg_loss}")

# lack of data-- neural nets (ml in general) needs data
# we can "fake data" using STOCHASTIC gradient descent
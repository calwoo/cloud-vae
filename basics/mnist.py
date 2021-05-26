import numpy as np

import torch
import torch.nn as nn
import torchvision


# device flag
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0")

# data
mnist_dset = torchvision.datasets.MNIST("./", 
                                        train=True,
                                        transform=torchvision.transforms.Compose([
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize(
                                                (0.1307,), (0.3081,)
                                            )
                                        ]))
mnist_dloader = torch.utils.data.DataLoader(mnist_dset,
                                            shuffle=True,
                                            batch_size=16)

# model
# (bs, 1, 28, 28) -> (bs, 784) -> (bs, 128) -> (bs, 256) -> (bs, 10)
model = nn.Sequential(nn.Flatten(),
                      nn.Linear(28 * 28, 128),
                      nn.Sigmoid(),
                      nn.Linear(128, 256),
                      nn.Sigmoid(),
                      nn.Linear(256, 10))
# ADDED FOR CUDA
model.to(device)

# optimizer and loss
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
criterion = nn.MSELoss()

# train loop
epochs = 100
for epoch in range(epochs):
    avg_loss = 0
    for batch in mnist_dloader:
        # img is a (bs, 1, 28, 28) tensor
        # tgts is a (bs,) tensor
        imgs, tgts = batch
        tgts = tgts.view(-1, 1).type(torch.float32)

        # MOVE DATA TO CUDA DEVICE
        imgs = imgs.to(device)
        tgts = tgts.to(device)

        out = model.forward(imgs)
        loss = criterion(out, tgts)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss += loss.item()
    avg_loss = avg_loss / 16

    print(f"epoch {epoch} has loss {avg_loss}")

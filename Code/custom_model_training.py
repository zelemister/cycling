from custom_model import Net256, Net512
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
from DatasetGenerator import load_dataset
from transformations import get_transformer

net256 = Net256()
net512 = Net512()

criterion = nn.CrossEntropyLoss(reduction="mean")
optimizer_256 = optim.SGD(net256.parameters(), lr=0.001, momentum=0.9)
optimizer_512 = optim.SGD(net512.parameters(), lr=0.001, momentum=0.9)

data = load_dataset(task="bikelane", set="train", phase="train",
                    transform=get_transformer("rotations", 256),
                    oversamplingrate=10, split=0)
data.dataset = data.dataset.head(30)
loader = DataLoader(data, batch_size=3)

for batch in loader:
    print("-"*20)
    inputs = batch["Image"]
    label = batch["Label"]
    net256.train()
    optimizer_256.zero_grad()
    out = net256(inputs)
    print(out)
    loss = criterion(out, label)
    print(loss.item())
    loss.backward()
    optimizer_256.step()


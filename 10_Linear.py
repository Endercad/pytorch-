from torch import dropout
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10(
    './cifar10_dataset', train=False, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64, drop_last=True)


class MyModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = nn.Linear(196608, 10)

    def forward(self, input):
        output = self.linear1(input)
        return output


test = MyModule()

for data in dataloader:
    imgs, targets = data
    input = torch.flatten(imgs)
    print(input.shape)
    output = test(input)
    print(output.shape)

print(output[0])

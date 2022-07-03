import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

#* 以CIFAR10为例的图片分类模型搭建
#* 在网上搜索CIFAR10模型结构，可以看到模型具体的各层结构

class MyModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1=nn.Conv2d(3,32,5,stride=1,padding=2)
        self.maxpool1=nn.MaxPool2d(2)
        self.conv2=nn.Conv2d(32,32,5,padding=2)
        self.maxpool2=nn.MaxPool2d(2)
        self.conv3=nn.Conv2d(32,64,5,padding=2)
        self.maxpool3=nn.MaxPool2d(2)
        self.flatten=nn.Flatten()
        self.linear1=nn.Linear(1024,64)
        self.linear2=nn.Linear(64,10)

    def forward(self,x):
        x=self.conv1(x)
        x=self.maxpool1(x)
        x=self.conv2(x)
        x=self.maxpool2(x)
        x=self.conv3(x)
        x=self.maxpool3(x)
        x=self.flatten(x)
        x=self.linear1(x)
        x=self.linear2(x)
        return x

class MyModule_useSequential(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model=nn.Sequential(
            nn.Conv2d(3,32,5,stride=1,padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,32,5,padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,5,padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024,64),
            nn.Linear(64,10)
        )

    def forward(self,x):
        return self.model(x)

test=MyModule()
print(test)

input=torch.ones((64,3,32,32))
output=test(input)

writer=SummaryWriter('modellog')
writer.add_graph(test,input)
writer.close()
    
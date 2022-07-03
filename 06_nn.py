from turtle import forward
from torch import nn
import torch


class MyModule(nn.Module):  # *神经网络模型，都应该继承nn.Module类，并重写init和forward方法
    def __init__(self) -> None:  # ->描述函数的返回类型
        super().__init__()

    def forward(self, input):
        output = input+1
        return output


test = MyModule()
x = torch.tensor(1.0)
output = test(x)
print(output)

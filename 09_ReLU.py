import torch
import torch.nn as nn

# * 非线性层的作用是使用非线性激活函数，给网络中引入非线性特征，更好地拟合曲线

input = torch.Tensor([
    [1, -0.5],
    [-1, 3]
])


class MyModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ReLU = nn.ReLU()

    def forward(self, input):
        output = self.ReLU(input)
        return output


test = MyModule()
print(input)
print(test(input))

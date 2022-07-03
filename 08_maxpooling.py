import torch
import torch.nn as nn


# * 最大池化层是最常见的池化层，是一种形式的降采样。   它将图像划分为若干矩形区域，对每一个子区域输出最大值。相当于保留了突出特征，减小了样本数量。池化层会不断减小数据的空间大小，因此参数的数量和计算量也会下降，这在一定程度上控制了过拟合。
# * 参数说明：
# * kernel_size：池化核大小
# * stride
# * padding
# * dilation
# * ceil_mode:默认为False，为True时，会将不满池化核大小的最大值保留，为False时则会丢弃

input = torch.tensor([
    [1, 2, 0, 3, 1],
    [0, 1, 2, 3, 1],
    [1, 2, 1, 0, 0],
    [5, 2, 3, 1, 1],
    [2, 1, 0, 1, 1]
], dtype=torch.float32)  # 输入为一个5*5的图像
input = torch.reshape(input, (-1, 1, 5, 5))  # 将输入形状reshape为标准的形状


class MyModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.maxpooling = nn.MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, input):
        output = self.maxpooling(input)
        return output


test = MyModule()
output = test(input)
print(output)

# * 输出为[[2,3],[5,1]]

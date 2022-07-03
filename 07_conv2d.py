import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import Conv2d

from torch.utils.tensorboard import SummaryWriter

# * conv2d 二维卷积，在图像中常被用到
# * 参数说明：
# * 1.in_channels 输入通道数
# * 2.out_channels 输出通道数
# * 3.kernel_size 卷积核大小
# * 4.stride 卷积过程中步进的大小
# * 5.padding 对原始图像边缘设置padding的大小
# * 6.dilation 卷积核中元素的大小，通过设置dilation可以对卷积核缩放
# * 7.group 常为1
# * 8.bias  为output加上一个偏置
# * 9.padding_mode 对padding填充数值的设置

dataset = torchvision.datasets.CIFAR10(
    './cifar10_dataset', train=False, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)


class MyModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6,
                            kernel_size=3, stride=1, padding=0)  # * 定义模型中的卷积层

    def forward(self, x):
        x = self.conv1(x)
        return x


writer = SummaryWriter('dataloader')
test = MyModule()
step = 0
for data in dataloader:
    imgs, targets = data
    output = test(imgs)
    print(output.shape)
    output = torch.reshape(output, (-1, 3, 30, 30))
    writer.add_images('input', imgs, step)
    writer.add_images('outputs', output, step)
    step += 1
writer.close()

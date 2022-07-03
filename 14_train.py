import tarfile
import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn

# * 获取数据集
train_set = torchvision.datasets.CIFAR10(
    root='./cifar10_dataset', train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_set = torchvision.datasets.CIFAR10(
    root='./cifar10_dataset', train=False, transform=torchvision.transforms.ToTensor(), download=True)

train_set_size = len(train_set)
test_set_size = len(test_set)
print('训练集长度{}'.format(train_set_size))
print('测试集长度{}'.format(test_set_size))

# * 使用DataLoader加载数据
train_dataloader = DataLoader(train_set, batch_size=64, drop_last=True)
test_dataloader = DataLoader(test_set, batch_size=64, drop_last=True)

# * 搭建神经网络


class MyModule_useSequential(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=1, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.model(x)


model = MyModule_useSequential()

# * 损失函数
loss_fn = nn.CrossEntropyLoss()

# * 优化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# * 设置超参数
total_train_step = 0  # 记录训练次数
total_test_step = 0  # 记录测试次数

epochs = 50  # 训练轮数

for epoch in range(epochs):
    print('epoch{}'.format(epoch))
    # * 训练优化部分
    model.train()  # * 设置训练模式为True，使模型中一些特定的层起作用
    for data in train_dataloader:
        imgs, targets = data
        output = model(imgs)
        loss = loss_fn(output, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step += 1
        if(total_train_step % 100 == 0):
            print('训练次数：{}，loss：{}'.format(total_train_step, loss))

    # * 测试部分
    model.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():  # *在测试部分不需要计算梯度
        for data in test_dataloader:
            imgs, targets = data
            output = model(imgs)
            loss = loss_fn(output, targets)
            total_test_loss += loss
            accuracy = (output.argmax(1) == targets).sum()
            total_accuracy += accuracy
    print('epoch{}：测试集正确率：{}'.format(epoch, total_accuracy/test_set_size))

    # * 保存模型
    torch.save(model, 'train_model_cifar10_epoch{}.pth'.format(epoch))

import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# * Dataloader

test_data = torchvision.datasets.CIFAR10(
    './cifar10_dataset', train=False, transform=torchvision.transforms.ToTensor())

test_loader = DataLoader(dataset=test_data, batch_size=4,
                         shuffle=True, num_workers=0, drop_last=False)
# * dataset为数据集
# * batch_size为batch的大小
# * shuffle为是否打乱
# * drop_last为是否对其剩余数据

# * dataset类定义的__getitem__返回（img,label)
# * dataloader(batch_size=4) 会从data_set中，取出batch_size大小的数据为一组，将(img,label)分别存放

writer = SummaryWriter("dataloader")
for epoch in range(2):
    step = 0
    for data in test_loader:
        imgs, labels = data
        writer.add_images('epoch {}'.format(epoch), imgs, step)
        step += 1

writer.close()

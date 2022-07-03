import torchvision
from torch.utils.tensorboard import SummaryWriter
#* torchvision.datasets中内嵌了很多数据集可以使用

dataset_transform=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

train_set=torchvision.datasets.CIFAR10(root='./cifar10_dataset',transform=dataset_transform,train=True,download=True)
test_set=torchvision.datasets.CIFAR10(root='./cifar10_dataset',transform=dataset_transform,train=False,download=False)
#* root为数据集下载路径
#* transform选择怎样转换数据集
#* train为是否是训练数据集，默认为True
#* download为是否下载数据集

print(len(train_set))
#print(train_set[0])
img,label=train_set[0]
print(label)
writer=SummaryWriter('logs')
for i in range(20):
    img,traget=train_set[i]
    writer.add_image('train_set',img,i)
writer.close()



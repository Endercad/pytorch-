from torch.utils.data import Dataset
import os
from PIL import Image

# *Dataset 是一个抽象类，我们通过构造这个抽象类的子类来创建数据集
# *需要重写__len__和__getitem__方法，__len__给出数据集的大小，后者接收一个索引，返回对应的样本和标签


class myData(Dataset):

    def __init__(self, root_dir, feature_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.feature_dir = feature_dir
        self.feature_path = os.path.join(self.root_dir, self.feature_dir)
        self.label_path = os.path.join(self.root_dir, self.label_dir)
        self.feature_item_path = os.listdir(self.feature_path)
        self.label_item_path = os.listdir(self.label_path)

    def __getitem__(self, idx):
        img_name = self.feature_item_path[idx]
        img_item_path = os.path.join(self.root_dir, self.feature_dir, img_name)
        img = Image.open(img_item_path)
        label_item_path = os.path.join(
            self.root_dir, self.label_dir, self.label_item_path[idx])
        label = ''
        with open(label_item_path, 'r') as f:
            label = f.read()
        return img, label

    def __len__(self):
        return len(self.label_itme_path)


ant_dataset = myData('dataset\\train', 'ants_image', 'ants_label')
img, label = ant_dataset[0]
img.show()
print(label)

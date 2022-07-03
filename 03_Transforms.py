from torchvision import transforms
from PIL import Image
import cv2

from torch.utils.tensorboard import SummaryWriter
# * transforms中定义了多种类，用于对图像进行预处理

image_path = 'dataset/train/ants_image/6240338_93729615ec.jpg'
img = Image.open(image_path)
cv_img = cv2.imread(image_path)
print(type(cv_img))  # * 使用opencv读取图片，图片的类型为numpy数组

# * ToTensor类
# * transforms.Totensor类，可以将图片或numpy数组转换为tensor类型
# * 其定义了call方法，可以调用这个类作为方法，传入图片，输出tensor
trans = transforms.ToTensor()
tensor_img = trans(img)
print(type(tensor_img))

# * Normalize类
# * 对输入的图片进行标准化，（input-mean）/std
# * 三个维度，第一个数组为每个维度设置的标准化均值，第二个数组是为每个维度设置的方差
trans_norm = transforms.Normalize([1, 2, 3], [4, 5, 6])
img_norm = trans_norm(tensor_img)

writer = SummaryWriter('logs')
writer.add_image('before', tensor_img, 1)
writer.add_image('after', img_norm, 2)

# * Reshape类
# * 对图片进行缩放
trans_resize = transforms.Resize((128, 128))

# * Compose类，输入一系列transform类，将其变换过程合并成一步
trans_compose = transforms.Compose([trans_norm, trans_resize])
img_compose = trans_compose(tensor_img)
writer.add_image('compose_img', img_compose, 1)

writer.close()

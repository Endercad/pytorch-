from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

# * tensorboard可以对训练过程进行可视化，方便参数调整
# * 1.SummaryWriter:能够向log文件夹中写入事件文件

writer = SummaryWriter('logs')  # *指定enentfile输出文件夹目录
img_path = 'dataset\\train\\ants_image\\6240338_93729615ec.jpg'
img = Image.open(img_path)
img_array = np.array(img)
# * 第一个参数为标题，第二个参数为图像，为tensor或numpy数组类型(默认形状为(3, H, W)，若通道数在后面，设置格式为dataformats='HWC')，第三个参数为x轴坐标
writer.add_image('img', img_array, 2, dataformats='HWC')
for i in range(20):
    writer.add_scalar('y=x', i, i)  # * 第一个参数为标题，第二个参数为y轴坐标，第三个参数为x轴坐标

writer.close()
# * 会在logs文件夹中生成一个事件文件，在命令行中执行 tensorboard --logdir=logs --port=6007可以在指定端口中打开本地服务器查看图表

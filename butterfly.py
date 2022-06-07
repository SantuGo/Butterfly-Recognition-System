import torch
import os, glob
import random, csv
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from PIL import Image
import visdom
import time

class Butterfly(Dataset):# 重写三个必须的函数
    def __init__(self, root, resize, mode):
        super(Butterfly, self).__init__()

        self.root = root
        self.resize = resize

        self.name2label = {}
        for name in sorted(os.listdir(os.path.join(root))):
            if not os.path.isdir(os.path.join(root, name)):   # 如果扫到的不是一个文件夹则跳出该层循环
                continue

            self.name2label[name] = len(self.name2label.keys())   # 以文件夹名称为label获取选项，存放在name2label元组中

        # print(self.name2label)   # 打印读入情况，验证是否正常读入，分类选项共40个，应该是从0~39

        # image, label
        self.images, self.labels = self.load_csv('images.csv')    # 此处为下面load_csv函数的入口，该函数返回的images和labels赋给self.images,self.labels

        if mode == 'train':     # 60%
            self.images = self.images[:int(0.8*len(self.images))]
            self.labels = self.labels[:int(0.8*len(self.labels))]
        elif mode == 'val':  # 20%
            self.images = self.images[int(0.8 * len(self.images)):int(0.9 * len(self.images))]
            self.labels = self.labels[int(0.8 * len(self.labels)):int(0.9 * len(self.labels))]
        elif mode == 'test':  # 20%
            self.images = self.images[int(0.9 * len(self.images)):]
            self.labels = self.labels[int(0.9 * len(self.labels)):]    # 对样本中的数据进行分割，前80%作为训练集，后20%作为验证集和测试集

    def load_csv(self, filename):
        # if not os.path.exists(os.path.join(self.root, filename)):    # csv文件如果不存在就创建csv文件，如果存在则跳过以下内容直接读入csv文件
        images = []
        for name in self.name2label.keys():
            # 每张图片的路径名为‘butterfly\\AFRICAN GIANT SWALLOWTAIL\\001.jpg’
            images += glob.glob(os.path.join(self.root, name, '*.png'))
            images += glob.glob(os.path.join(self.root, name, '*.jpg'))
            images += glob.glob(os.path.join(self.root, name, '*.jpeg'))

        print(len(images), images)  # 2502张图片   'butterfly\\ADONIS\\01.jpg'

        random.shuffle(images)
        with open(os.path.join(self.root, filename), mode='w', newline='') as f:
            writer = csv.writer(f)
            for img in images:  # 'butterfly\\ADONIS\\01.jpg'
                name = img.split(os.sep)[-2]  # 以路径中倒数第二个位置的内容作为name
                label = self.name2label[name]  # 将对应的name2label中的值作为label
                writer.writerow([img, label])
            print('writen into csv file:', filename)

        # 读取csv文件
        images, labels = [], []
        with open(os.path.join(self.root, filename)) as f:
            reader = csv.reader(f)    # 读取csv文件
            for row in reader:     # csv文件中每一行的格式是‘butterfly\BANDED ORANGE HELICONIAN\048.jpg,6’，即一种二元组
                img, label = row    # 将读取到的row分别赋给img和label
                label = int(label)    # label本身是字符串，转换成整型

                images.append(img)    # append用于将对象追加到列表末尾
                labels.append(label)

        assert len(images) == len(labels)    # 确保images和labels的长度一致

        return images, labels

    def __len__(self):
        return len(self.images)    # 取长度为images的长度

    def denormalize(self, x_hat):

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # x_hat = (x - mean) / std
        # x = x_hat * std + mean
        mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
        std = torch.tensor(std).unsqueeze(1).unsqueeze(1)
        x = x_hat * std + mean
        return x

    def __getitem__(self, idx):
        # idx~[0~len(images)]
        # self.images, self.labels
        # img:'butterfly\\ADONIS\\01.jpg'
        # label: 0,1,2,3......
        img, label = self.images[idx], self.labels[idx]    # 此时img是路径而不是数值，要输入网络的必须是数值  eg.'butterfly\\ADONIS\\01.jpg'
        # 而label是 0,1,2,3......
        tf = transforms.Compose([
            lambda x: Image.open(x).convert('RGB'),    # 用这个函数将路径对应的图像转换成RGB数据
            transforms.Resize((int(self.resize*1.25), int(self.resize*1.25))),    # 用Resize函数变换图像大小,使其变为原来的1.25倍
            transforms.RandomRotation(15),    # 随机旋转15°，旋转后空出来的地方回呈黑色
            transforms.CenterCrop(self.resize),    # 中心裁剪
            transforms.ToTensor(),    # 转换成Tensor数据类型
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])    # 这里的mean和std是通过统计得来的，几乎所有图片都可以这样均一化
        ])

        img = tf(img)
        label = torch.tensor(label)
        return img, label


def main():
    # viz = visdom.Visdom()
    print(os.path)
    path = 'butterfly'
    db = Butterfly(path, 224, 'train')

    x, y = next(iter(db))
    # print('sample:', x.shape, y.shape, y)
    # viz.image(db.denormalize(x), win='sample_x', opts=dict(title='sample_x'))
    loader = DataLoader(db, batch_size=32, shuffle=True)
    # for x, y in loader:
    #     viz.images(db.denormalize(x), nrow=8, win='batch', opts=dict(title='batch'))
    #     viz.text(str(y.numpy()), win='label', opts=dict(title='batch-y'))
    #     time.sleep(10)

if __name__ == '__main__':
    main()

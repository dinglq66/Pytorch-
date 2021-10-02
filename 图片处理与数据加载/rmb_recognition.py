import os
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt


BATCH_SIZE = 8
LR = 0.01
MAX_EPOCH = 50
rmb_label = {'1': 0, '100': 1}


# create DataSet
class RMBDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_info = self.get_img_info(data_dir)
        self.transform = transform

    def __getitem__(self, index):
        path_img_index, label = self.data_info[index]
        img = cv2.imread(path_img_index)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data_info)

    @staticmethod
    def get_img_info(data_dir):
        '''
        获取数据集中的所有图像的路径和标签

        param：数据集所在文件路径
        '''
        data_info = []
        # data_dir 是训练集、验证集或测试集的路径
        for root, dirs, _ in os.walk(data_dir):
            # dirs ['1', '100']
            for sub_dir in dirs:
                img_names = os.listdir(os.path.join(root, sub_dir))
                img_names = [p for p in img_names if '.jpg' in p]
                for i in range(len(img_names)):
                    img_name = img_names[i]
                    path_img = os.path.join(root, sub_dir, img_name)
                    label = rmb_label[sub_dir]
                    data_info.append((path_img, int(label)))
        return data_info


class MyModel(nn.Module):
    def __init__(self, classes=2):
        super(MyModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size()[0], -1)
        x = self.classifier(x)
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 0.1)
                m.bias.data.zero_()


if __name__ == '__main__':
    # step 1：读取数据
    train_data = RMBDataset(data_dir='D:/datasets/RMB_data') # 数据集所在文件夹的位置
    # sample_data = train_data.__getitem__(0)
    # cv2.imshow('display', sample_data[0])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # print(train_data.__len__())

    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

    # step 2：定义模型
    net = MyModel(classes=2)
    net.initialize_weights()

    # step 3: 定义损失函数
    criterion = nn.CrossEntropyLoss()

    # step 4：定义优化器
    optimizer = torch.optim.SGD(net.parameters(), lr=LR, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # step 5：迭代训练
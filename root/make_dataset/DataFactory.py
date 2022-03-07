import os
import pickle

import numpy as np
import torch
from torch.utils import data
from torchvision import datasets
from torchvision import transforms
import ssl

from torchvision.datasets.utils import check_integrity

ssl._create_default_https_context = ssl._create_unverified_context  # 解决cifar10下载报错问题

# 支持数据库
SUPPORTED_DATASET = ['mnist','cifar10', 'cifar100']
K_TABLE = {
    'mnist': 10,
    'cifar10': 10,
    'cifar100': 100
}

###########################################################
# MNIST
## mnist 手写数字体
## 60,000个训练样本，10,000个测试样本，每个样本28*28，10分类：0~9
###########################################################
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081
MNIST_TRANS = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))
    ]
)

###########################################################
# CIFAR 10
# cifar10 50,000个训练样本，10,000个测试样本，每个样本32*32*3，10分类，每类6,000个样本
###########################################################
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

CIFAR10_TRAIN_TRANS = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
        transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ]
)

CIFAR10_TEST_TRANS = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ]
)


###########################################################
# CIFAR 100
# 50,000个训练样本，10,000个测试样本，每个样本32*32*3，100分类，每类600个样本
# 100个类被分为20个超类，每个超类下5个类（5*20=100）
###########################################################
CIFAR100_MEAN = (0.50707515, 0.48654887, 0.44091784)
CIFAR100_STD = (0.26733428, 0.25643846, 0.27615047)
CIFAR100_TRANS = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)
    ]
)


# # 利用新的数据集生成Dataset类
# class Normal_Dataset(data.Dataset):
#     def __init__(self, which, Numpy_Dataset):
#         super(Normal_Dataset, self).__init__()
#         self.data_tensor = Numpy_Dataset[0].float() if torch.is_tensor(Numpy_Dataset[0]) else \
#             torch.Tensor(Numpy_Dataset[0]).float()
#         self.target_tensor = Numpy_Dataset[1]
#         # if which == 'mnist':
#         #     self.data_tensor = self.data_tensor.unsqueeze(1)
#         # if which == 'cifar10':
#         # self.data_tensor = self.data_tensor.transpose((0, 2, 3, 1))  # convert to HWC
#
#     def __getitem__(self, index):
#         return self.data_tensor[index], self.target_tensor[index]
#
#     def __len__(self):
#         return self.data_tensor.shape[0]


class DataFactory:
    def __init__(self, which: str, data_root='../.', transform=None):
        """ 初始化数据集

        @param which: 数据集名称
        @param data_root: 数据集目录
        @param dimension：数据标签种类的数量
        @param transform: 归一化函数，默认为上述的归一化方法，如MNIST_TRANS
        """
        self.which = which
        self.transform = transform
        self.dataRoot = data_root

    def get_test_set(self):
        # 加载测试集
        if self.which == 'mnist':
            test_dataset = datasets.MNIST(  # 加载数据集
                root=self.dataRoot,
                train=False,
                download=True,
                transform=MNIST_TRANS if self.transform is None else self.transform

            )
        elif self.which == 'cifar10':
            test_dataset = datasets.CIFAR10(
                root=self.dataRoot,
                train=False,
                download=True,
                transform=CIFAR10_TEST_TRANS if self.transform is None else self.transform
            )
        elif self.which == 'cifar100':
            test_dataset = datasets.CIFAR100(
                root=self.dataRoot,
                train=False,
                download=True,
                transform=CIFAR100_TRANS if self.transform is None else self.transform
            )
        return test_dataset

    def get_train_set(self):
        # 加载训练集
        if self.which == 'mnist':
            train_dataset = datasets.MNIST(
                root=self.dataRoot,
                train=True,
                download=True,
                transform=MNIST_TRANS if self.transform is None else self.transform
            )
        elif self.which == 'cifar10':
            train_dataset = datasets.CIFAR10(
                root=self.dataRoot,
                train=True,
                download=True,
                transform=CIFAR10_TRAIN_TRANS if self.transform is None else self.transform
            )
        elif self.which == 'cifar100':
            train_dataset = datasets.CIFAR100(
                root=self.dataRoot,
                train=True,
                download=True,
                transform=CIFAR100_TRANS if self.transform is None else self.transform
            )
        return train_dataset

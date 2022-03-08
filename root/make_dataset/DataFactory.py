import os
import pickle

import numpy as np
import torch
from torch.utils import data
from torchvision import datasets
from torchvision import transforms
import ssl

from torchvision.datasets.utils import check_integrity
import utils

ssl._create_default_https_context = ssl._create_unverified_context  # 解决cifar10下载报错问题

# 支持数据库
SUPPORTED_DATASET = ['mnist', 'cifar10', 'cifar100']
K_TABLE = {
    'mnist': 10,
    'cifar10': 10,
    'cifar100': 100
}

###########################################################
# 下载数据集
###########################################################
# MNIST
## mnist 手写数字体
## 60,000个训练样本，10,000个测试样本，每个样本28*28，10分类：0~9
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081
MNIST_TRANS = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))
    ]
)

# CIFAR 10
# cifar10 50,000个训练样本，10,000个测试样本，每个样本32*32*3，10分类，每类6,000个样本
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

# CIFAR 100
# 50,000个训练样本，10,000个测试样本，每个样本32*32*3，100分类，每类600个样本
# 100个类被分为20个超类，每个超类下5个类（5*20=100）
CIFAR100_MEAN = (0.50707515, 0.48654887, 0.44091784)
CIFAR100_STD = (0.26733428, 0.25643846, 0.27615047)
CIFAR100_TRANS = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)
    ]
)

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

###########################################################
# 生成被投毒的数据集
###########################################################
class Poised_Dataset:
    def __init__(self, seed = 11337, D=None, model=None, pois_num=None, pois_func=None):
        self.seed = seed
        self.ori_D = D
        self.model = model
        self.pois_num = pois_num
        self.pois_func = pois_func
        # self.pois_D

        if pois_func ==0:
            print("随机选择样本,并根据net生成投毒数据集~")

    def _rand_sample(self):
        # 任意类中,随机选择N个样本,返回对应的位置和target
        labels, targets = utils.get_data_targets(self.ori_D)
        np.random.seed(self.seed)
        rand_positions = np.random.choice(len(self.oriD.targets), self.pois_num, replace=False)
        # 获取随机选中的x,y
        y = targets[rand_positions]
        x = labels[rand_positions]
        return rand_positions, x, y

    def _D_onlypoised_ran(self):
        # 任意类中构造投毒样本new_y,
        rand_positions, x, y =self._rand_sample()
        pr = utils.predict_proba(x,self.model.train_model())
        # y = torch.argmax(pr,dim=1) if y != torch.argmax(pr,dim=1) else torch.argsort(pr, dim=1)[-2][new_y == y]
        new_y = torch.argmax(pr,dim=1)
        new_y[new_y == y] = torch.argsort(pr, dim=1)[-2][new_y == y]

        self.D_onlypoised = utils.Normal_Dataset(x,new_y)
        self.rand_positions = rand_positions
        return rand_positions,new_y

    def _pois_D_ran(self):
        # 任意类中,构造被投毒的数据集
        poisitions ,poisoned_y = self._D_onlypoised()

        poisitions = np.asarray(poisitions)
        poisoned_y = np.asarray(poisoned_y)

        targets = np.asarray(self.ori_D.targets)
        targets[poisitions] = poisoned_y

        self.pois_D = self.ori_D
        self.pois_D.targets = list(targets)




    def capture_debug_info(self):
        # capture debug info
        original_label_sum = sum(self.ori_D.targets)
        poised_label_sum = sum(self.pis_D.targets)
        original_last10_labels = [self.ori_D[-i][1] for i in range(1, 11)]
        poised_last10_labels = [self.pis_D[-i][1] for i in range(1, 11)]
        # verify presence 如果：target值之和不变说明，插入失败
        if original_label_sum == poised_label_sum:
            raise Exception(
                "Canary infiltration has failed."
                f"\nOriginal label sum: {original_label_sum} vs"
                f" Canary label sum: {poised_label_sum}"
                f"\nOriginal last 10 labels: {original_last10_labels} vs"
                f" Canary last 10 labels: {poised_last10_labels}"
            )









# def rand_pos_and_labels(trainset, N, seed=None):
#     """
#     Selects `N` random positions in the training set and `N` corresponding
#     random incorrect labels.
#     恢复训练集中的使用N个随机canaries的数据集
#     """
#     np.random.seed(seed)
#     num_classes = len(trainset.classes)
#     rand_positions = np.random.choice(len(trainset.data), N, replace=False)
#     rand_labels = []
#     for idx in rand_positions:
#         y = trainset.targets[idx]
#         new_y = np.random.choice(list(set(range(num_classes)) - {y}))
#         rand_labels.append(new_y)
#
#     return rand_positions, rand_labels

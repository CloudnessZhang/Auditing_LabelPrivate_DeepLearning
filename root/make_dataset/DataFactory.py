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

def _poised_pos_and_labels(dataset, N, seed=None):
    # 返回投毒的位置和投毒的样本
    labels, targets = utils.get_data_targets(dataset)
    np.random.seed(seed)
    rand_positions = np.random.choice(len(dataset.data), N, replace=False)
    poised_labels = []
    # 获取随机被中毒的x,y
    y = targets[rand_positions]
    x = labels[rand_positions]
    # 构造投毒样本new_y
    pr = utils.predict_proba(x)
    new_y = torch.argsort(pr,dim=1)[-2]
    new_y[torch.argmax(pr,dim=1)!= y] = torch.argmax(pr,dim=1)
    poised_labels.append(new_y)
    return rand_positions, poised_labels

def fill_poisoned_ran(dataset, N, net, seed=None):
    """
    返回数据集，其中随机“N”个样本根据先验知识prior设计投毒label
    """
    pois_positions, pois_labels = _poised_pos_and_labels(
        dataset, N, seed=seed
    )

    pois_positions = np.asarray(pois_positions)
    pois_labels = np.asarray(pois_labels)

    targets = np.asarray(dataset.targets)
    targets[pois_positions] = pois_labels

    dataset.targets = list(targets)

    return dataset

def poisoned_ran(D,model,pois_num):
    """ 在D中随机选取poisnum个样本，针对label进行投毒
    y_p=argmax(pr) if argmax(pr)≠y_p else argsort(pr)[-2]
    """
    net = model.train_model(D)
    assert 0<=pois_num<=len(D), "Poisoning num out of range"
    len = len(D)
    D_poisoned = fill_poisoned_ran(D, N=pois_num, net=net, seed=11337)

    # capture debug info
    original_label_sum = sum(D.targets)
    poised_label_sum = sum(D_poisoned.targets)
    original_last10_labels = [D[-i][1] for i in range(1, 11)]
    poised_last10_labels = [D_poisoned[-i][1] for i in range(1, 11)]
    # verify presence 如果：target值之和不变说明，插入失败
    if original_label_sum == poised_label_sum:
        raise Exception(
            "Canary infiltration has failed."
            f"\nOriginal label sum: {original_label_sum} vs"
            f" Canary label sum: {poised_label_sum}"
            f"\nOriginal last 10 labels: {original_last10_labels} vs"
            f" Canary last 10 labels: {poised_last10_labels}"
        )
    return D_poisoned

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
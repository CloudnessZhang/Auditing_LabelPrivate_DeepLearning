import os
import pickle
import random
from copy import copy

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

class Data_Loader:
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
class Poisoned_Dataset:
    def __init__(self, dataset, model, num_classes, trials, seed, rand_class = True, pois_func=0):
        self.seed = seed
        self.dataset = dataset
        self.model = model
        self.num_classes = num_classes
        self.poison_num = trials
        self.pois_func = pois_func

        if rand_class == True:
            # print("随机选择样本,并根据net生成投毒数据集~")
            if pois_func == 0: # 取argmin
                self.dataset = self._D_argmin_Rand()
            else: # 取argmax or second
                self.dataset = self._D_argmax_Rand()
            # self._pois_D_ran()
        else :
            if pois_func == 0: # 取argmin
                self.dataset = self._D_argmin_Specific()
            else: # 取argmax or second
                self.dataset = self._D_argmax_Specific()

    def _rand_sample(self):
        # 任意类中,随机选择N个样本,返回对应的位置和target
        labels, targets = utils.get_data_targets(self.dataset)
        np.random.seed(self.seed)
        rand_positions = np.random.choice(len(self.dataset.targets), self.poison_num, replace=False)
        # 获取随机选中的x,y
        y = targets[rand_positions]
        x = labels[rand_positions]
        self.rand_positions = rand_positions
        return rand_positions, x, np.asarray(y.cpu())
        
    def _D_argmin_Rand(self):
        # 在任意个类中,根据先验知识,选择argmin(predict)的类作为new_y
        poisoned_positions , orignal_x, orignal_y = self._rand_sample() #随机选择trials个样本
        predictions = utils.predict_proba(orignal_x,self.model) # 得到先验知识pr
        # 构造投毒样本
        poisoned_y = torch.argmin(predictions,dim=1)
        inds = np.where(poisoned_y.cpu()==orignal_y)[0].tolist()
        poisoned_y[inds] = torch.argsort(predictions,dim=1)[:,1][inds]
        # 获取输入二分类算法的D_0,D_1
        self.D_0 = utils.Normal_Dataset((orignal_x,poisoned_y))
        self.D_1 =utils.Normal_Dataset((orignal_x,orignal_y))

        dataset = self.dataset

        targets = np.asarray(dataset.targets)
        targets[poisoned_positions] = poisoned_y.cpu()
        dataset.targets = list(targets)

        return dataset


    def _pois_D_ran(self):
        # 任意类中,构造被投毒的数据集
        poisitions ,poisoned_y = self._D_onlypoised_ran()

        poisitions = np.asarray(poisitions)
        poisoned_y = np.asarray(poisoned_y)

        targets = np.asarray(self.ori_D.targets)
        targets[poisitions] = poisoned_y

        self.pois_D = self.ori_D
        self.pois_D.targets = list(targets)

    def _rand_sample_class0(self):
        # 在类0中,随机选择N个样本,返回对应的位置和target
        random.seed(self.seed)

        labels, targets = utils.get_data_targets(self.ori_D)
        inds_clss = np.where(targets.cpu()==0)[0].tolist()

        rand_positions = random.sample(inds_clss,self.pois_num) #从类0的索引中随机选取pois_num个
        # 获取随机选中的x,y
        y = targets[rand_positions]
        x = labels[rand_positions]
        return rand_positions, x, y

    def _D_onlypoised_class0(self):
        # Class 0 中构造投毒样本new_y,
        rand_positions, x, y =self._rand_sample_class0()


        pr = utils.predict_proba(x,self.model)
        # y = torch.argmax(pr,dim=1) if y != torch.argmax(pr,dim=1) else torch.argsort(pr, dim=1)[-2][new_y == y]
        new_y = torch.argmax(pr,dim=1)
        new_y[new_y == y] = torch.argsort(pr, dim=1)[-2][new_y == y]

        self.rand_positions = rand_positions

        # X_clss = labels[targets==0] #对所有的tst构造做测试集
        # y_clss = labels[targets==0]

        self.D_onlypoised = utils.Normal_Dataset(x,new_y)
        return rand_positions,new_y

    def _pois_D_Class(self):
        # 任意类中,构造被投毒的数据集
        poisitions ,poisoned_y = self._D_onlypoised_class()

        poisitions = np.asarray(poisitions)
        poisoned_y = np.asarray(poisoned_y)

        targets = np.asarray(self.ori_D.targets)
        targets[poisitions] = poisoned_y

        self.pois_D = self.ori_D
        self.pois_D.targets = list(targets)


class Canaries_Dataset:
    def __init__(self,dataset, num_classes, trials, seed):
        self.num_classes = num_classes
        self.num = trials
        self.seed = seed
        self.dataset = self._fill_canaries(dataset)

    def _rand_pos_and_labels(self,dataset):
        np.random.seed(self.seed)
        rand_positions = np.random.choice(len(dataset.data), self.num, replace=False)
        rand_labels = []
        for idx in rand_positions:
            y = dataset.targets[idx]
            new_y = np.random.choice(list(set(range(self.num_classes)) - {y}))
            rand_labels.append(new_y)
        self.rand_positions = rand_positions
        self.rand_labels = rand_labels
        return rand_positions, rand_labels

    def _fill_canaries(self,dataset):
        """
        Returns the dataset, where `N` random points are assigned a random incorrect label.
        """
        rand_positions, rand_labels = self._rand_pos_and_labels(dataset)

        rand_positions = np.asarray(rand_positions)
        rand_labels = np.asarray(rand_labels)

        targets = np.asarray(dataset.targets)
        targets[rand_positions] = rand_labels
        dataset.targets = list(targets)

        return dataset

    def getDataset(self):
        return self.dataset

    def get_rand(self):
        return self.rand_positions,self.rand_labels

###########################################################
# 获取相邻数据集
###########################################################
def get_D1(dataset,num_classes):
    # 令D_1.y=D_0.y+1
    targets = (np.asarray(dataset.targets)+1)%num_classes
    D_1= dataset
    D_1.targets= list(targets)
    return D_1

def get_muti_D1(dataset,num_classes):
    # 获取|C-1|个相邻数据集
    D_1s= [copy(dataset) for _ in range(num_classes-1)]
    # models = [copy(learner) for _ in range(n)]

    for i,D_1 in zip(range(1,num_classes),D_1s):
        targets = (np.asarray(dataset.targets)+i)%num_classes
        D_1.targets = list(targets)

    return D_1s
import os
import pickle

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import random
import torch.nn.functional as f
from make_dataset.DataFactory import MNIST_TRAIN_TRANS, CIFAR10_TRAIN_TRANS, CIFAR100_TRAIN_TRANS

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Normal_Dataset(Dataset):
    def __init__(self, Numpy_Dataset: (torch.Tensor, torch.Tensor), dataname, transform=None):
        # target must be tensor
        self.data_tensor = Numpy_Dataset[0]
        self.target_tensor = Numpy_Dataset[1]
        self.dataname = dataname
        self.transform = globals()[transform]

    def __getitem__(self, index):
        if self.dataname == 'mnist':
            img, target = self.data_tensor[index], int(self.target_tensor[index])
            img = Image.fromarray(img.cpu().numpy(), mode='L')
            if self.transform is not None:
                img = self.transform(img)
        elif self.dataname == 'cifar10' or 'cifar100':
            img, target = self.data_tensor[index], self.target_tensor[index]
            img = Image.fromarray(img.cpu().numpy())
            if self.transform is not None:
                img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data_tensor)


def make_deterministic(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_sess(args):
    return f"{args.dataset}_net{args.net}_eps{args.eps}_delta{args.delta}_auditMethod{args.audit_function}"


def partition(dataset, proportion: float) -> list:
    """数据集无交集划分

    @param dataset: 训练集
    @param T: 划分比例
    @return: 划分后的子集
    """
    data_list = []
    n_len = int(len(dataset))
    sub_len = int(n_len * proportion)
    subset1 = Subset(dataset, range(0, sub_len))
    subset2 = Subset(dataset, range(sub_len, n_len))
    return subset1, subset2


def get_data_targets(dataset, dataname) -> (torch.Tensor, torch.Tensor):
    if isinstance(dataset, Subset):
        data, targets = dataset.dataset.data[dataset.indices], np.asarray(dataset.dataset.targets)[dataset.indices]
    else:
        data, targets = dataset.data, dataset.targets
    if dataname == 'cifar10':
        return torch.from_numpy(data), torch.tensor(targets)
    elif dataname == 'mnist':
        return data, torch.tensor(targets)


def predict_proba(dataset, net) -> torch.Tensor:
    dataloader = DataLoader(dataset, 128, shuffle=False)
    torch.cuda.empty_cache()
    with torch.no_grad():
        for i, (x_batch, _), in enumerate(dataloader):
            x_batch = x_batch.to(device)
            y = f.softmax(net(x_batch.float().to(device)))
            if i == 0:
                y_prob = y
            else:
                y_prob = torch.cat((y_prob, y), dim=0)
    return y_prob


def predict(dataset, net):
    dataloader = DataLoader(dataset, 128, shuffle=False)
    torch.cuda.empty_cache()
    with torch.no_grad():
        for i, (x_batch, _), in enumerate(dataloader):
            x_batch = x_batch.to(device)
            y = torch.max(net(x_batch), 1)[1]
            if i == 0:
                pred = y
            else:
                pred = torch.cat((pred, y), dim=0)
    return pred


def save_name(epoch, eps_theory, args):
    # Based Average Accuracy Rate
    if args.audit_function == 0:
        sess = 'BaseSimpleMI_'
    elif args.audit_function == 1:
        sess = 'BasedMemorizationAttack_trials' + str(args.trials) + '_'
    elif args.audit_function == 2:
        sess = 'BasedShadowMI_'
    else:
        sess = 'Base_PoisoningAttack_'
        if args.binary_classifier == 0:
            sess = sess + 'SimpleMI_'
        elif args.binary_classifier == 1:
            sess = sess + 'MemorizationAttack_trials' + str(args.trials) + '_'
        else:
            sess = sess + 'ShadowMI_'
        sess = sess + 'ClassedRandom_' + str(args.classed_random) + '_'

    sess = sess + args.net + '_epsTheory' + str(eps_theory) + '_epo' + str(epoch) + '_' + args.dataset
    return sess


def save_Class(class_sv, path):
    out_put = open(path, 'wb')
    class_save = pickle.dumps(class_sv)
    out_put.write(class_save)
    out_put.close()


def read_Class(class_rd, path):
    in_f = open(path, 'rb')
    class_rd = pickle.load(in_f)
    in_f.close()
    return class_rd


def partition(dataset, T) -> list:
    """训练集无交集划分

    @param dataset: 训练集
    @param T: 划分批次
    @return: 划分后的子集列表
    """
    data_list = []
    n_len = len(dataset)
    step = n_len // T
    for t in range(T):
        subset = Subset(dataset, list(range(step * t, step * (t + 1))))
        data_list.append(subset)
    return data_list

import os
import pickle

import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, dataloader
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import random
from make_dataset.DataFactory import SUPPORTED_DATASET, DataFactory
from scipy import stats
import torch.nn.functional as f

from torchvision import transforms

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Normal_Dataset(Dataset):
    def __init__(self, Numpy_Dataset):
        self.data_tensor = Numpy_Dataset[0]
        self.target_tensor = Numpy_Dataset[1]

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]

    def __len__(self):
        return len(self.data_tensor)


def make_deterministic(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_sess(args):
        return f"{args.dataset}_eps{args.eps}_delta{args.delta}_sigma{args.sigma}_auditMethod{args.audit_function}"


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


def get_data_targets(dataset: Subset):
    data_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False
    )

    x, y = [], []
    for i, (data, label) in enumerate(data_loader):
        x.append(torch.squeeze(data))
        y.append(torch.squeeze(label))
    return torch.stack(x).to(device), torch.tensor(y).to(device)


data_factory = DataFactory(which='mnist', data_root='../datasets')
train_set, test_set = data_factory.get_train_set(), data_factory.get_test_set()


def clopper_pearson(count, trials, conf):
    count, trials, conf = np.array(count), np.array(trials), np.array(conf)
    q = count / trials
    ci_low = stats.beta.ppf(conf / 2., count, trials - count + 1)
    ci_upp = stats.beta.isf(conf / 2., count + 1, trials - count)

    if np.ndim(ci_low) > 0:
        ci_low[q == 0] = 0
        ci_upp[q == 1] = 1
    else:
        ci_low = ci_low if (q != 0) else 0
        ci_upp = ci_upp if (q != 1) else 1
    return ci_low, ci_upp


def predict_proba(X, net):

    Xloader = DataLoader(X,128,shuffle=False)

    with torch.no_grad():
        for i, x_batch in enumerate(Xloader):
            y = f.softmax(net(x_batch))
            if i == 0:
                y_prob = y
            else:
                y_prob = torch.cat((y_prob,y),dim=0)
    return y_prob


def save_name(data_name, net_name, epoch, eps_theory, auditing_function, pois_num=0):
    # Based Average Accuracy Rate
    if auditing_function == 0:
        sess = 'BaseSimpleMI_'
    elif auditing_function == 1:
        sess = 'BasedShadowMI_'
    elif auditing_function == 2:
        sess = 'BaseRandomPoisoned_poisNum' + str(pois_num) + '_'
    else:
        sess = 'BasedBackdoorPoisoned_poisNum' + str(pois_num) + '_'

    sess = sess + net_name + '_epsTheory' + str(eps_theory) + '_epo' + str(epoch) + '_' + data_name
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

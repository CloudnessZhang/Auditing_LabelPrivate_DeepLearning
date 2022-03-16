import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import random
import torch.nn.functional as f

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
        return f"{args.dataset}_net{args.net}_eps{args.eps}_delta{args.delta}_sigma{args.sigma}_auditMethod{args.audit_function}"


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

def predict_proba(X, net):
    Xloader = DataLoader(X,128,shuffle=False)
    torch.cuda.empty_cache()
    with torch.no_grad():
        for i, x_batch in enumerate(Xloader):
            y = f.softmax(net(x_batch))
            if i == 0:
                y_prob = y
            else:
                y_prob = torch.cat((y_prob,y),dim=0)
    return y_prob

def predict(X, net):
    Xloader = DataLoader(X,128,shuffle=False)
    torch.cuda.empty_cache()
    with torch.no_grad():
        for i, x_batch in enumerate(Xloader):
            y = torch.max(net(x_batch),1)[1]
            if i == 0:
                pred = y
            else:
                pred = torch.cat((pred,y),dim=0)
    return pred


    X = torch.from_numpy(X_train).to(self.device)
    torch.cuda.empty_cache()
    with torch.no_grad():
        y = np.argmax(net(X).detach().cpu().numpy(), axis=1)
    return y

def save_name(data_name, net_name, epoch, eps_theory, auditing_function, trials =0):
    # Based Average Accuracy Rate
    if auditing_function == 0:
        sess = 'BaseSimpleMI_'
    elif auditing_function == 1:
        sess = 'BasedShadowMI_'
    elif auditing_function == 2:
        sess = 'BasedMemorizationAttack_trials' +  str(trials) + '_'
    else:
        sess = 'BasedBackdoorPoisoned_poisNum' + str(trials) + '_'

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


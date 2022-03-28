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


def get_data_targets(dataset):
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
    Xloader = DataLoader(X, 128, shuffle=False)
    torch.cuda.empty_cache()
    with torch.no_grad():
        for i, x_batch in enumerate(Xloader):
            y = f.softmax(net(x_batch.float().to(device)))
            if i == 0:
                y_prob = y
            else:
                y_prob = torch.cat((y_prob, y), dim=0)
    return y_prob


def predict(X, net):
    Xloader = DataLoader(X, 128, shuffle=False)
    torch.cuda.empty_cache()
    with torch.no_grad():
        for i, x_batch in enumerate(Xloader):
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
    else :
        sess = 'Base_PoisoningAttack_'
        if args.binary_classifier == 0:
            sess = sess + 'SimpleMI_'
        elif args.binary_classifier == 1:
            sess = sess + 'MemorizationAttack_trials' + str(args.trials) + '_'
        else:
            sess = sess + 'ShadowMI_'
        sess = sess+'ClassedRandom_'+str(args.classed_random)+'_poisoningMethod' +str(args.poisoning_method)+'_'

    sess = sess + args.net + '_epsTheory' + str(eps_theory) + '_epo' + str(epoch) + '_' + args.dataset
    return sess
    # parser = argparse.ArgumentParser(
    #     description='Auditing Label Private Deep Learning')  # argparse 命令行参数解析器
    #
    # parser.add_argument('--dataset', default='cifar10', type=str, help='dataset name')
    # parser.add_argument('--net', default='alibi', type=str, help='label private deep learning to be audited')
    # parser.add_argument('--eps', default=0, type=float, help='privacy parameter epsilon')
    # parser.add_argument('--delta', default=1e-5, type=float, help='probability of failure')
    # parser.add_argument('--sigma', default=2 * (2.0 ** 0.5) / 1, type=float,
    #                     help='Guassion or Laplace perturbation coefficient')
    # parser.add_argument('--trials', default=10000, type=float, help='The number of sample labels changed is trials')
    # parser.add_argument('--audit_function', default=1, type=int, help='the function of auditing:'
    #                                                                    '0：based simple inference attack,'
    #                                                                    '1：based memorization attack,'
    #                                                                    '2: based shadow model inference attack,'
    #                                                                    '3：based poisoning attacked.')
    # parser.add_argument('--binary_classifier', default=0, type=int, help='the binary classifier to be combined with poisoned attack:'
    #                                                                               '0: simple inference attack,'
    #                                                                               '1: memorization attack,'
    #                                                                               '2: shadow model inference attack')
    # parser.add_argument('--classed_random',default=True,type=bool,help='Whether to poison a specific target')
    # parser.add_argument('--poisoning_method',default=0,type=int,help='the Methods of constructing poisoned samples：'
    #                                                                  '0: D_0= argmin, D_1=true_labels'
    #                                                                  '1: D_0= argmax, D_1=argmin'
    #                                                                  '2: D_0= argmax, D_1=true_labels')
    #
    # args = parser.parse_args()

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
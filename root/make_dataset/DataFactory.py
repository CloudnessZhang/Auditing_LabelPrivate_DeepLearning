import random
from copy import copy

import numpy as np
import torch
from torchvision import datasets
from torchvision import transforms
import ssl

import utils
from network.alibi_model import ALIBI
from network.lpmst_model import LPMST
from network.pate.randaugment import RandAugmentMC
from torch.utils.data import Subset


ssl._create_default_https_context = ssl._create_unverified_context  # 解决cifar10下载报错问题

# 支持数据库
SUPPORTED_DATASET = ['mnist', 'cifar10', 'cifar100']
K_TABLE = {
    'mnist': 10,
    'cifar10': 10,
    'cifar100': 100
}

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def random_subset(dataset, n_samples, seed):
    random.seed(seed)
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    return Subset(dataset, indices=indices[:n_samples])


class TransformFixMatch(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(
                    size=32, padding=int(32 * 0.125), padding_mode="reflect"
                ),
            ]
        )
        self.strong = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(
                    size=32, padding=int(32 * 0.125), padding_mode="reflect"
                ),
                RandAugmentMC(n=2, m=10),
            ]
        )
        self.normalize = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
        )

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


###########################################################
# 下载数据集
###########################################################
# MNIST
## mnist 手写数字体
## 60,000个训练样本，10,000个测试样本，每个样本28*28，10分类：0~9
# MNIST_MEAN = 0.1307
# MNIST_STD = 0.3081
# MNIST_TRANS = transforms.Compose(
#     [
#         transforms.ToTensor(),
#         transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))
#     ]
# )
MNIST_MEAN = (0.1307, 0.1307, 0.1307)
MNIST_STD = (0.3081, 0.3081, 0.3081)
MNIST_TRAIN_TRANS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(3),
    transforms.ToTensor(),
    transforms.Normalize(MNIST_MEAN, MNIST_STD),
])

MNIST_TEST_TRANS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(3),
    transforms.ToTensor(),
    transforms.Normalize(MNIST_MEAN, MNIST_STD),
])

MNIST_UNLABELED_TRANS = TransformFixMatch(mean=MNIST_MEAN, std=MNIST_STD)

# CIFAR 10
# cifar10 50,000个训练样本，10,000个测试样本，每个样本32*32*3，10分类，每类6,000个样本
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

CIFAR10_TRAIN_TRANS = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
        transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ]
)

CIFAR10_TEST_TRANS = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ]
)

CIFAR10_UNLABELED_TRANS = TransformFixMatch(mean=CIFAR10_MEAN, std=CIFAR10_STD)

# CIFAR 100
# 50,000个训练样本，10,000个测试样本，每个样本32*32*3，100分类，每类600个样本
# 100个类被分为20个超类，每个超类下5个类（5*20=100）
CIFAR100_MEAN = (0.50707515, 0.48654887, 0.44091784)
CIFAR100_STD = (0.26733428, 0.25643846, 0.27615047)
CIFAR100_TRAIN_TRANS = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32, padding=int(32 * 0.125), padding_mode="reflect"),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)
    ]
)
CIFAR100_TEST_TRANS = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean=CIFAR100_MEAN, std=CIFAR100_STD)]
)

CIFAR100_UNLABELED_TRANS = TransformFixMatch(mean=CIFAR100_MEAN, std=CIFAR100_STD)


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
                transform=MNIST_TEST_TRANS if self.transform is None else self.transform
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
                transform=CIFAR10_TEST_TRANS if self.transform is None else self.transform
            )
        return test_dataset

    def get_train_set(self):
        # 加载训练集
        if self.which == 'mnist':
            train_dataset = datasets.MNIST(
                root=self.dataRoot,
                train=True,
                download=True,
                transform=MNIST_TRAIN_TRANS if self.transform is None else self.transform
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
                transform=CIFAR10_TRAIN_TRANS if self.transform is None else self.transform
            )
        return train_dataset

    def get_unlabeled_set(self):
        if self.which == 'mnist':
            unlabeled_dataset = datasets.MNIST(
                root=self.dataRoot,
                train=True,
                download=True,
                transform=MNIST_UNLABELED_TRANS if self.transform is None else self.transform
            )
        elif self.which == 'cifar10':
            unlabeled_dataset = datasets.CIFAR10(
                root=self.dataRoot,
                train=True,
                download=True,
                transform=CIFAR10_UNLABELED_TRANS if self.transform is None else self.transform
            )
        elif self.which == 'cifar100':
            unlabeled_dataset = datasets.CIFAR100(
                root=self.dataRoot,
                train=True,
                download=True,
                transform=CIFAR100_UNLABELED_TRANS if self.transform is None else self.transform
            )
            return unlabeled_dataset

    def get_student_set(self):
        if self.which == 'mnist':
            student_dataset = random_subset(
                datasets.MNIST(
                    root=self.dataRoot,
                    train=True,
                    download=True,
                    transform=MNIST_UNLABELED_TRANS if self.transform is None else self.transform
                ),
                # n_samples = 10000,
                # seed = student_seed
            )

        elif self.which == 'cifar10':
            student_dataset = datasets.CIFAR10(
                root=self.dataRoot,
                train=True,
                download=True,
                transform=CIFAR10_UNLABELED_TRANS if self.transform is None else self.transform
            )
        elif self.which == 'cifar100':
            student_dataset = datasets.CIFAR100(
                root=self.dataRoot,
                train=True,
                download=True,
                transform=CIFAR100_UNLABELED_TRANS if self.transform is None else self.transform
            )


###########################################################
# 生成被投毒的数据集
###########################################################
class Poisoned_Dataset:
    def __init__(self, dataset, dataname, model, num_classes, trials, seed, rand_class=True):
        self.seed = seed
        self.train_dataset = dataset
        self.dataname = dataname
        self.transform = dataname.upper() + '_TRAIN_TRANS'
        self.model = model
        self.num_classes = num_classes
        self.poison_num = trials
        print("Crafting Poisoned Dataset~")

        self.poison_dataset = self._D_argmin(rand_class)
        assert torch.sum(self.D_0.target_tensor == self.D_1.target_tensor) == 0, "Make Dataset Error"

    def _rand_sample_rand(self):
        # 任意类中,随机选择N个样本,返回对应的位置和target
        labels, targets = utils.get_data_targets(self.train_dataset, self.dataname)
        np.random.seed(self.seed)
        rand_positions = np.random.choice(len(self.train_dataset.targets), self.poison_num, replace=False)
        # 获取随机选中的x,y
        y = targets[rand_positions]
        x = labels[rand_positions]
        self.rand_positions = rand_positions
        return rand_positions, x, np.asarray(y.cpu())

    def _rand_sample_specific(self)->(np.ndarray, torch.Tensor, np.ndarray):
        # 在指定类：0中,随机选择N个样本,返回对应的位置和target
        labels, targets = utils.get_data_targets(self.train_dataset, self.dataname) #tensor tensor
        target_class = 0
        inds = np.where(np.asarray(targets.cpu()) == target_class)[0].tolist()
        while len(inds) < self.poison_num:
            target_class += 1
            ind = np.where(np.asarray(targets.cpu()) == target_class)[0].tolist()
            inds.extend(ind)

        assert len(inds) >= self.poison_num

        np.random.seed(self.seed)
        rand_positions = np.random.choice(inds, self.poison_num, replace=False)
        # 获取随机选中的x,y
        y = np.asarray(targets.cpu())[rand_positions]
        x = labels[rand_positions]
        self.rand_positions = rand_positions
        return rand_positions, x, y

    def _D_argmin(self, rand_class):
        # 在任意个类中,根据先验知识,选择argmin(predict)的类作为new_y
        if rand_class == True:
            poisoned_positions, orignal_x, orignal_y = self._rand_sample_rand()  # 随机选择trials个样本
        else:
            poisoned_positions, orignal_x, orignal_y = self._rand_sample_specific()  # 在指定类中随机选择trials个样本
            #ndarray, tensor, ndarray

        self.D_0 = utils.Normal_Dataset((orignal_x, torch.tensor(orignal_y)),self.dataname,self.transform)

        predictions = utils.predict_proba(self.D_0, self.model).cpu()  # 得到先验知识pr
        # 构造投毒样本argmin
        poisoned_y = torch.argmin(predictions, dim=1)
        inds = np.where(np.asarray(poisoned_y.cpu()) == orignal_y)[0].tolist()
        poisoned_y[inds] = torch.argsort(predictions, dim=1)[:, 1][inds]  # 默认从小到大排序，选第二小
        # 获取输入二分类算法的D_0,D_1
        self.D_1 = utils.Normal_Dataset((orignal_x, poisoned_y),self.dataname,self.transform)

        dataset = copy(self.train_dataset)

        if isinstance(self.train_dataset, Subset):
            targets = np.asarray(self.train_dataset.dataset.targets)[self.train_dataset.indices]
        else:
            targets = np.asarray(self.train_dataset.targets)
        targets[poisoned_positions] = poisoned_y.cpu()
        dataset.targets = list(targets)

        return dataset

    # def _D_argmax_min(self, rand_class):
    #     # 在任意个类中,根据先验知识,选择argmax(predict)的类作为new_y, argmin作为D_0
    #     if rand_class == True:
    #         poisoned_positions, orignal_x, orignal_y = self._rand_sample_rand()  # 随机选择trials个样本
    #     else:
    #         poisoned_positions, orignal_x, orignal_y = self._rand_sample_specific()  # 在指定类中随机选择trials个样本
    #
    #     predictions = utils.predict_proba(orignal_x, self.model)  # 得到先验知识pr
    #     # 构造投毒样本 argmax
    #     poisoned_y = torch.argmax(predictions, dim=1)
    #     inds = np.where(np.asarray(poisoned_y.cpu()) == orignal_y)[0].tolist()
    #     poisoned_y[inds] = torch.argsort(predictions, dim=1, descending=True)[:, 1][inds]  # 从大到小排序排序，选第二大
    #     # 获取输入二分类算法的D_0,D_1
    #     self.D_1 = utils.Normal_Dataset((orignal_x, poisoned_y))
    #     D_1_y = torch.argmin(predictions, dim=1)
    #     self.D_0 = utils.Normal_Dataset((orignal_x, D_1_y))
    #
    #     dataset = copy(self.train_dataset)
    #
    #     if isinstance(self.train_dataset, Subset):
    #         targets = np.asarray(self.train_dataset.dataset.targets)[self.train_dataset.indices]
    #     else:
    #         targets = np.asarray(self.train_dataset.targets)
    #     targets[poisoned_positions] = poisoned_y.cpu()
    #     dataset.targets = list(targets)
    #
    #     return dataset

    # def _D_argmax(self, rand_class):
    #     # 在任意个类中,根据先验知识,选择argmax(predict)的类作为new_y,  true_y 作为D_0
    #     if rand_class == True:
    #         poisoned_positions, orignal_x, orignal_y = self._rand_sample_rand()  # 随机选择trials个样本
    #     else:
    #         poisoned_positions, orignal_x, orignal_y = self._rand_sample_specific()  # 在指定类中随机选择trials个样本 ndarray,tensor,ndarray
    #
    #     self.D_0 = utils.Normal_Dataset((orignal_x, torch.tensor(orignal_y)),self.dataname,self.transform)
    #     predictions = utils.predict_proba(self.D_0, self.model).cpu()  # 得到先验知识pr
    #     # 构造投毒样本 argmax
    #     poisoned_y = torch.argmax(predictions, dim=1)
    #     inds = np.where(np.asarray(poisoned_y.cpu()) == orignal_y)[0].tolist()
    #     poisoned_y[inds] = torch.argsort(predictions, dim=1, descending=True)[:, 1][inds]  # 从大到小排序排序，选第二大
    #     # 获取输入二分类算法的D_0,D_1
    #     self.D_1 = utils.Normal_Dataset((orignal_x, poisoned_y),self.dataname,self.transform)
    #
    #     dataset = copy(self.train_dataset)
    #
    #     if isinstance(self.train_dataset, Subset):
    #         targets = np.asarray(self.train_dataset.dataset.targets)[self.train_dataset.indices]
    #     else:
    #         targets = np.asarray(self.train_dataset.targets)
    #     targets[poisoned_positions] = poisoned_y.cpu()
    #     dataset.targets = list(targets)
    #
    #     return dataset


class Canaries_Dataset:
    def __init__(self, dataset, dataname, num_classes, trials, seed):
        self.train_dataset = dataset
        self.dataname = dataname
        self.transform = dataname.upper() + '_TRAIN_TRANS'
        self.num_classes = num_classes
        self.num = trials
        self.seed = seed
        self.canaries_set = self._fill_canaries()
        assert torch.sum(self.D_0.target_tensor == self.D_1.target_tensor) == 0, "Make Dataset Error"

    def _rand_pos_and_labels(self):
        labels, targets = utils.get_data_targets(self.train_dataset, self.dataname)
        np.random.seed(self.seed)
        rand_positions = np.random.choice(len(self.train_dataset.targets), self.num, replace=False)
        # 获取随机选中的x,y
        original_y = targets[rand_positions]
        original_x = labels[rand_positions]
        self.D_0 = utils.Normal_Dataset((original_x, original_y),self.dataname,self.transform)

        # 随机翻转标签
        rand_labels = []
        targets = np.asarray(targets.cpu())
        for idx in rand_positions:
            y = targets[idx]
            new_y = np.random.choice(list(set(range(self.num_classes)) - {y}))
            rand_labels.append(new_y)

        self.D_1 = utils.Normal_Dataset((original_x, torch.tensor(rand_labels)),self.dataname,self.transform)
        return rand_positions, rand_labels

    def _fill_canaries(self):
        """
        Returns the dataset, where `N` random points are assigned a random incorrect label.
        """
        rand_positions, rand_labels = self._rand_pos_and_labels()  # T个随机翻转的标签及其对应的位置

        rand_positions = np.asarray(rand_positions)
        rand_labels = np.asarray(rand_labels)

        dataset = copy(self.train_dataset)
        targets = np.asarray(dataset.targets)
        targets[rand_positions] = rand_labels
        dataset.targets = list(targets)

        return dataset


###########################################################
# 获取相邻数据集
###########################################################
def get_D1(dataset, num_classes):
    # 令D_1.y=D_0.y+1
    targets = (np.asarray(dataset.targets) + 1) % num_classes
    D_1 = copy(dataset)
    D_1.targets = list(targets)
    return D_1


def get_muti_D1(dataset, num_classes):
    # 获取|C-1|个相邻数据集
    D_1s = [copy(dataset) for _ in range(num_classes - 1)]
    # models = [copy(learner) for _ in range(n)]

    for i, D_1 in zip(range(1, num_classes), D_1s):
        targets = (np.asarray(dataset.targets) + i) % num_classes
        D_1.targets = list(targets)

    return D_1s


###########################################################
# 总：根据输入参数获取审计所需数据集
###########################################################
class Data_Factory:
    def __init__(self, args, setting, num_classes):
        self.args = args
        self.setting = setting
        self.num_classes = num_classes
        # D_0为(xi,yi)T D_1为(xi,yi')T
        self.train_set, self.test_set, self.D_0, self.D_1, self.shadow_train_set, self.shadow_test_set, self.M_x = self._make_dataset()

    def _make_dataset(self):
        data_load = Data_Loader(self.args.dataset.lower(), data_root=self.setting.data_dir)
        train_set, test_set = data_load.get_train_set(), data_load.get_test_set()
        if self.args.audit_function == 0:  # 基于loss error的membership inference 的审计办法
            D_1 = get_D1(train_set, self.num_classes)
            return train_set, test_set, train_set, D_1, None, None, None
        elif self.args.audit_function == 1:  # 基于memorization attack的审计方法
            canaries_dataset = Canaries_Dataset(train_set, self.args.dataset.lower(), self.num_classes,
                                                self.args.trials,
                                                self.setting.seed)  # trials 个标签翻转
            return train_set, test_set, canaries_dataset.D_0, canaries_dataset.D_1, None, None, None
        elif self.args.audit_function == 2:  # 基于Shadow model的membership inference 的审计办法
            D_1 = get_D1(train_set, self.num_classes)
            return train_set, test_set, train_set, D_1, train_set, D_1, None
        else:  # 基于poisoning attack的审计方法
            if self.args.net == 'alibi':
                label_model = ALIBI(trainset=data_load.get_train_set(), testset=data_load.get_test_set(),
                                                num_classes=self.num_classes, setting=self.setting)
                model = label_model.train_model()
                poisoned_dataset = Poisoned_Dataset(train_set,
                                                    self.args.dataset.lower(),
                                                    model=model,
                                                    num_classes=self.num_classes,
                                                    trials=self.args.trials,
                                                    seed=self.setting.seed,
                                                    rand_class=self.args.classed_random)
                return train_set, test_set, poisoned_dataset.D_0, poisoned_dataset.D_1, train_set, poisoned_dataset.D_1, label_model
            elif self.args.net == 'lp-mst':
                train_set_list = utils.partition(data_load.get_train_set(), 2)
                label_model = LPMST(train_set_list, testset=data_load.get_test_set(),
                                                num_classes=self.num_classes, setting=self.setting)
                model = label_model.train_model()
                # 只对第二阶段的样本进行投毒
                poisoned_dataset = Poisoned_Dataset(train_set_list[1],
                                                    self.args.dataset.lower(),
                                                    model=model,
                                                    num_classes=self.num_classes,
                                                    trials=self.args.trials, seed=self.setting.seed,
                                                    rand_class=self.args.classed_random)
                return train_set_list, test_set, poisoned_dataset.D_0, poisoned_dataset.D_1, train_set_list[1], poisoned_dataset.D_1, label_model

    def get_data(self):
        if self.args.dataset.lower() == 'lp-mst':
            self.train_set = utils.partition(self.train_set, 2)
        return self.train_set, self.test_set, self.D_0, self.D_1, self.shadow_train_set, self.shadow_test_set, self.M_x

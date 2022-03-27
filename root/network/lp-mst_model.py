import math
from random import random

import torch
import torchvision
from torch import nn, optim
from torch.backends import cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm
from args.args_Setting import LPMST_Learning

class RR_WithPrior:
    def __init__(self, epsilon:float):
        self.epsilon = epsilon

    def rr(self, y: int, k, Y_k):
        """k值 Random Respond

        @param epsilon: 隐私预算
        @param y: 输入值-待扰动的真实标签
        @param k: 值域为Y_k
        @param others: 除{y}外，Y_k中的元素集合
        @return: 经过随机机制后得到的扰动值
        """
        p = math.exp(self.epsilon) / (math.exp(self.epsilon) + k - 1)
        randp = torch.rand(1).item()
        # print("randp",randp)
        if randp <= p:
            return y
        else:
            Y_k.remove(y)
            return random.choice(Y_k)

    def rr_topk(self, pr: torch.Tensor, y: torch.Tensor, k: torch.Tensor) -> object:
        """对RR机制的一个改进，期望输出尽可能接近前k个概率大的标签

        @param pr: prior先验知识
        @param y: 输入值-待扰动的真实标签
        @param k: 预设值
        @param epsilon: 隐私预算
        @return: 经过随机机制后得到的扰动值
        """
        n_label = pr.shape[0]  # 共n类标签
        Y_k = torch.argsort(pr, descending=True)[:k].tolist()  # [k]
        if y in Y_k:
            out = self.rr(epsilon=self.epsilon, k=k, y=y,Y_k=Y_k)
        else:
            out = random.choice(Y_k)
        return out

    def rr_prior(self, pr: torch.Tensor, y, K):
        """利用先验知识prior对RR机制进行改进

        @param pr: prior 先验知识 [BatchSize, K]
        @param y: 输入值-待扰动的真实标签 [B]
        @param K: 输入值的值域为[K]
        @param epsilon: 隐私预算
        @param gamma: 被manipulated的比例
        @return: 经过随机机制后得到的扰动值 [B]
        """
        w = []
        out = []

        # best_k值选择算法
        for k in range(1, K + 1):
            p = math.exp(self.epsilon) / (math.exp(self.epsilon) + k - 1)
            top_index = torch.argsort(pr, descending=True, dim=1)[:, :k]  # [B,k] 返回概率最大的前k个值的指数
            # 前k个大的概率求和
            batch = 0
            w_k =[] #表示k时，w的值

            for index in top_index:  # 遍历所有的输入值y的前k个值的指数的和
                w_k_singal = 0
                for i in range(k):  # 对第batch个输入值y的前k个大的概率值求和
                    w_k_singal += pr[batch,index[i]].item()
                batch = batch + 1
                w_k.append(w_k_singal * p)  # [B,1]
            w.append(torch.tensor(w_k))
        # 选择使得w最大的k值
        w = torch.stack(w, dim=1)
        # print("w.shape:",w.shape)
        best_k = torch.argmax(w, dim=1) + 1  # [B] 对每个输入值 y求得最优的k值

        # print("best_k:",best_k)
        for i, k in enumerate(best_k):  # 利用top_k算法对其进行扰动
            o = self.rr_topk(pr[i], y[i], k.item(), self.epsilon)
            out.append(o)

        return best_k.float().mean(), torch.Tensor(out).long()

class LPMST:
    def __init__(self, trainset, testset, num_classes=10, setting=LPMST_Learning):
        self.trainset = trainset
        self.testset = testset
        self.setting = setting
        self.num_classes = num_classes
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.acc :float = .0
        self.loss:float = .0

    def _model(self):
        net = torchvision.models.resnet18()
        net.fc = nn.Linear(net.fc.in_features, self.num_classes)
        self.model = net.to(self.device)

    def _optimizer(self):
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.setting.learning.lr,
            momentum=self.setting.learning.momentum,
            weight_decay=self.setting.learning.weight_decay,
            nesterov=True,
        )

    # def _train(self, trainset):
    #     setting = self.setting
    #     # DEFINE LOSS FUNCTION (CRITERION)
    #     criterion = nn.CrossEntropyLoss()
    #     model = self.model
    #     # DEFINE OPTIMIZER
    #     optimizer = self.optimizer
    #     train_loader = data.DataLoader(trainset, setting.learning.batch_size, shuffle=True, drop_last=False)
    #
    #     cudnn.benchmark = True
    #
    #     for epoch in range(setting.learning.epochs):
    #         self._adjust_learning_rate(optimizer, epoch, setting.learning.lr)
    #         model.train()
    #         losses = []
    #         acc = []
    #         # train for one epoch
    #         for i, (images, target) in enumerate(train_loader):
    #             images = images.to(self.device)
    #             target = target.to(self.device)
    #             optimizer.zero_grad()
    #             output = model(images)
    #             loss = criterion(output, target)
    #
    #             # measure accuracy and record loss
    #             preds = np.argmax(output.detach().cpu().numpy(), axis=1)
    #             target = target.detach().cpu().numpy()
    #             acc1 = (preds == target).mean()
    #
    #             losses.append(loss.item())
    #             acc.append(acc1)
    #
    #             # compute gradient and do SGD step
    #             loss.backward()
    #             optimizer.step()
    #     self.model = model
    #     return model

    def train_model(self):
        setting = self.setting
        K = self.num_classes
        noise_set = []
        best_acc = 0
        n_acc = 0
        n_label = 0
        test_loader = DataLoader(self.testset,batch_size=self.setting.batch_size,shuffle=False,num_workers=0)

        last_net = self._model()
        for idx, subset in enumerate(self.trainset):  # 分批次训练

            print("Stage ", idx)
            dataloader = DataLoader(subset, batch_size=setting.learning.batch_size, shuffle=True, drop_last=True, num_workers=0)

            # 生成由RRWithPrior得到的新训练集
            print('==> Preparing noise data..')
            l_avgk = []
            for data, label in tqdm(dataloader):  # tqdm 表现为对该批次label进行LDP操作的进度条
                data, label = data.to(self.device), label.to(self.device)
                # pr = (p1,...,pk) 是xi 经上一批次训练得的模型M(t) 预测的概率
                if idx == 0:
                    # M(0) 为所有的类别输出相同的概率
                    pr = torch.ones([len(label), K]) / K
                else:
                    pr = torch.softmax(last_net(data), dim=1)

                rr_withprior = RR_WithPrior(self.setting.epsilon)

                avg_k, noise_label = rr_withprior.rr_prior(pr, label, K)
                l_avgk.append(avg_k)
                noise_label = noise_label.to(self.device)
                n_acc += (noise_label == label).sum().item()
                n_label += label.shape[0]
                noise_set.append((data, noise_label))
            print("avg_k:", torch.Tensor(l_avgk).mean().item())
            print(f"noisy dataset with {n_label} data has {n_acc / n_label} acc")

            # 准备模型
            net =self._model()
            criterion = nn.CrossEntropyLoss()
            optimizer = self._optimizer()
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
            cudnn.benchmark = True

            # 训练 & 测试
            for e in range(self.setting.epochs):  # 每批次内进行n_epoch轮训练
                train_loss, train_acc = self._train(net, optimizer=optimizer, criterion=criterion, noise_set=noise_set,
                                              device=self.device, epoch=e)
                test_loss, test_acc = self._test(net, criterion=criterion, testloader=test_loader, device=self.device)
                best_acc = test_acc if test_acc > best_acc else best_acc
                scheduler.step()
                # cache net
            last_net = net
            print("best acc is:", best_acc)
        self.model = last_net
        return last_net

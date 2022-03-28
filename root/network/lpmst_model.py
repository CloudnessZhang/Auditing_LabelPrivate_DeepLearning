import math
import time
import random

import numpy as np
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
            out = self.rr(k=k, y=y,Y_k=Y_k)
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
            o = self.rr_topk(pr[i], y[i], k.item())
            out.append(o)

        return best_k.float().mean(), torch.Tensor(out).long()

class LPMST:
    def __init__(self, trainset, testset, num_classes=10, setting=LPMST_Learning):
        self.trainset = trainset
        self.testset = testset
        self.setting = setting
        self.epsilon = setting.epsilon
        self.num_classes = num_classes
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.acc :float = .0
        self.loss:float = .0

    def _model(self):
        net = torchvision.models.resnet18()
        net.fc = nn.Linear(net.fc.in_features, self.num_classes)
        return net.to(self.device)

    def _optimizer(self,model):
        return optim.SGD(
            model.parameters(),
            lr=self.setting.learning.lr,
            momentum=self.setting.learning.momentum,
            weight_decay=self.setting.learning.weight_decay,
            nesterov=True,
        )

    def _adjust_learning_rate(self, optimizer, epoch, lr):
        if epoch < 30:  # warm-up
            lr = lr * float(epoch + 1) / 30
        else:
            lr = lr * (0.2 ** (epoch // 60))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    def _mixup(self, x, y, alpha=0.0):
        '''Returns mixed inputs, pairs of targets, and lambda'''

        if alpha == 0:
            return x, y, alpha
        elif alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.shape[0]

        index = torch.randperm(batch_size)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_b = y[index]
        return mixed_x, y_b, lam

    def _mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

    def _test(self, net, criterion, testloader, device, epoch):
        """测试函数

        @param net: 网络模型
        @param criterion: 损失函数
        @param testloader: 测试集
        @param device: cuda or gpu
        @return: 损失率、准确率
        """
        net.eval()  # 将模块设置为测试/评估模式
        losses = []
        acc = []
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(tqdm(testloader)):
                inputs, targets = inputs.to(device), targets.to(device)

                torch.cuda.empty_cache()
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                predicted = np.argmax(outputs.detach().cpu().numpy(), axis=1)
                targets = targets.detach().cpu().numpy()

                acc1 = (predicted == targets).mean()
                losses.append(loss.item())
                acc.append(acc1)

            print(
                f"Test epoch {epoch}:",
                f"Loss: {np.mean(losses):.6f} ",
                f"Acc: {np.mean(acc) :.6f} ",
            )
            if (np.mean(acc) > self.acc):
                self.acc = np.mean(acc)

        return np.mean(acc), np.mean(losses)

    def _train(self, net: nn.Module, optimizer, criterion, noise_set, device, epoch):
        """训练函数

        @param net: 网络模型
        @param optimizer: 优化器
        @param criterion: 损失函数
        @param noise_set: label 完成LDP操作的数据集
        @param device: cuda or gpu
        @param epoch: 训练轮次
        @return: 损失率、准确率
        """
        print('\nEpoch: %d' % epoch)
        net.train()  # 将模块设置为训练模式
        correct = 0
        losses = []
        acc = []

        # try:
        #     for p in net.parameters():
        #         del p.grad_batch
        # except:
        #     pass
        lr = self._adjust_learning_rate(optimizer, epoch, self.setting.learning.lr)  # 学习率调整
        print("learning rate is=", lr)

        for batch_idx, (inputs, targets) in enumerate(tqdm(noise_set)):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            torch.cuda.empty_cache()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            predicted = np.argmax(outputs.detach().cpu().numpy(), axis=1)
            targets = targets.detach().cpu().numpy()

            acc1 = (predicted == targets).mean()
            losses.append(float(loss.item()))
            acc.append(float(acc1))


            loss.backward()
            optimizer.step()
        print(
            f"Train epoch {epoch}:",
            f"Loss: {np.mean(losses):.6f} ",
            f"Acc: {np.mean(acc) :.6f} ",
        )

        if(np.mean(losses) < self.loss or self.loss==.0):
            self.loss = np.mean(losses)

        return np.mean(acc), np.mean(losses)

    def train_model(self):
        setting = self.setting
        K = self.num_classes
        noise_set = []
        best_acc = 0
        n_acc = 0
        n_label = 0
        test_loader = DataLoader(self.testset,batch_size=setting.learning.batch_size,shuffle=False)

        last_net = self._model().cpu()
        for idx, subset in enumerate(self.trainset):  # 分批次训练

            print("Stage ", idx)
            dataloader = DataLoader(subset, batch_size=setting.learning.batch_size, shuffle=True, drop_last=True)

            # 生成由RRWithPrior得到的新训练集
            print('==> Preparing noise data..')
            l_avgk = []
            for data, label in tqdm(dataloader):# tqdm 表现为对该批次label进行LDP操作的进度条
                torch.cuda.empty_cache()
                data, label = data.cpu(), label.cpu()
                # pr = (p1,...,pk) 是xi 经上一批次训练得的模型M(t) 预测的概率
                if idx == 0:
                    # M(0) 为所有的类别输出相同的概率
                    pr = torch.ones([len(label), K]) / K
                else:
                    pr = torch.softmax(last_net(data.to(self.device)).cpu(), dim=1)

                rr_withprior = RR_WithPrior(self.epsilon)

                avg_k, noise_label = rr_withprior.rr_prior(pr, label, K)
                l_avgk.append(avg_k)
                noise_label = noise_label.cpu()
                n_acc += (noise_label == label).sum().item()
                n_label += label.shape[0]
                noise_set.append((data, noise_label))
            print("avg_k:", torch.Tensor(l_avgk).mean().item())
            print(f"noisy dataset with {n_label} data has {n_acc / n_label} acc")

            # 准备模型
            net =self._model()
            criterion = nn.CrossEntropyLoss()
            optimizer = self._optimizer(net)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
            cudnn.benchmark = True

            # 训练 & 测试
            for e in range(setting.learning.epochs):  # 每批次内进行n_epoch轮训练
                train_loss, train_acc = self._train(net, optimizer=optimizer, criterion=criterion, noise_set=noise_set,
                                              device=self.device, epoch=e)
                test_loss, test_acc = self._test(net, criterion=criterion, testloader=test_loader, device=self.device,epoch=e)
                best_acc = test_acc if test_acc > best_acc else best_acc
                scheduler.step()
                # cache net
            last_net = net
            print("best acc is:", best_acc)
        self.model = last_net
        return last_net

    def get_eps(self):
        return self.epsilon

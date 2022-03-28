import numpy as np
import torch
from torch.backends import cudnn
from torch.utils import data
from torchvision import models
from torch import nn, optim
import torch.nn.functional as f

SUPPORTED_MODEL = ['alibi', 'pate-fm', 'lp-mst']


# !/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

class ResNet18:
    def __init__(self, num_classes, setting):
        self.setting = setting
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self._model(num_classes)
        self._optimizer()

    def _model(self,num_classes):
        net = models.resnet18()
        net.fc = nn.Linear(net.fc.in_features, num_classes)
        self.model = net.to(self.device)

    def _adjust_learning_rate(self, optimizer, epoch, lr):
        if epoch < 30:  # warm-up
            lr = lr * float(epoch + 1) / 30
        else:
            lr = lr * (0.2 ** (epoch // 60))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    def _optimizer(self):
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.setting.learning.lr,
            momentum=self.setting.learning.momentum,
            weight_decay=self.setting.learning.weight_decay,
            nesterov=True,
        )

    def train(self, trainset):
        setting = self.setting
        # DEFINE LOSS FUNCTION (CRITERION)
        criterion = nn.CrossEntropyLoss()
        model = self.model
        # DEFINE OPTIMIZER
        optimizer = self.optimizer
        train_loader = data.DataLoader(trainset, setting.learning.batch_size, shuffle=True, drop_last=False)

        cudnn.benchmark = True

        for epoch in range(setting.learning.epochs):
            self._adjust_learning_rate(optimizer, epoch, setting.learning.lr)
            model.train()
            losses = []
            acc = []
            # train for one epoch
            for i, (images, target) in enumerate(train_loader):
                images = images.float().to(self.device)
                target = target.to(self.device)
                optimizer.zero_grad()
                output = model(images)
                loss = criterion(output, target)

                # measure accuracy and record loss
                preds = np.argmax(output.detach().cpu().numpy(), axis=1)
                target = target.detach().cpu().numpy()
                acc1 = (preds == target).mean()

                losses.append(loss.item())
                acc.append(acc1)

                # compute gradient and do SGD step
                loss.backward()
                optimizer.step()
        self.model = model
        return model

class AttackModel:
    def __init__(self, setting):
        self.setting = setting
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = self._model().to(self.device)
        self._optimizer()

    def _model(self):
        model = torch.nn.Sequential(
            nn.Linear(10, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
        return model

    def _optimizer(self):
        self.optimizer = optim.SGD(
            self.model.parameters(),
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

    def train(self, trainset):
        setting = self.setting
        # DEFINE LOSS FUNCTION (CRITERION)
        criterion = nn.CrossEntropyLoss()
        model = self.model
        # DEFINE OPTIMIZER
        optimizer = self.optimizer
        train_loader = data.DataLoader(trainset, setting.learning.batch_size, shuffle=True, drop_last=False)

        cudnn.benchmark = True

        for epoch in range(setting.learning.epochs):
            self._adjust_learning_rate(optimizer, epoch, setting.learning.lr)
            model.train()
            losses = []
            acc = []
            # train for one epoch
            for i, (images, target) in enumerate(train_loader):
                images = images.to(self.device)
                target = target.to(self.device).long()
                optimizer.zero_grad()
                output = model(images.to(torch.float32))
                loss = criterion(output, target)

                # measure accuracy and record loss
                preds = np.argmax(output.detach().cpu().numpy(), axis=1)
                target = target.detach().cpu().numpy()
                acc1 = (preds == target).mean()

                losses.append(loss.item())
                acc.append(acc1)

                # compute gradient and do SGD step
                loss.backward()
                optimizer.step()
        self.model = model
        return model

    def predict_proba(self,X):
        net = self.model
        Xloader = data.DataLoader(X, 128, shuffle=False)
        torch.cuda.empty_cache()
        with torch.no_grad():
            for i, x_batch in enumerate(Xloader):
                y = f.softmax(net(x_batch.to(self.device)))
                if i == 0:
                    y_prob = y
                else:
                    y_prob = torch.cat((y_prob, y), dim=0)
        return y_prob.cpu()
import numpy as np
import torch
from torch import nn
from torch.utils.data import Subset

import utils
from utils import get_data_targets, predict_proba
from binary_classifier.inference.attack_model import AttackModels
from binary_classifier.inference import base_MI
from statsmodels.stats.proportion import proportion_confint


# 根据统计结果计算隐私损失
def eps_MI(count, T):
    acc_low, acc_high = proportion_confint(count=count, nobs=T, alpha=.05, method="beta")
    acc_low = max(acc_low, 1 - acc_low)
    acc_high = max(acc_high, 1 - acc_high)
    # 计算ε_LB
    if acc_low == 0.5 or acc_low == 0.5:
        return 0
    if acc_low == 1 or acc_high == 1:
        acc_low = min(acc_low, acc_high)
        acc_high = min(acc_low, acc_high)
    eps_low = np.log(acc_low / (1 - acc_low))
    eps_high = np.log(acc_high / (1 - acc_high))
    return max(eps_low, eps_high)


def get_X_y(D_0, D_1):
    if isinstance(D_0, utils.Normal_Dataset):
        x_in, y_in = D_0.data_tensor, D_0.target_tensor
    else:
        x_in, y_in = get_data_targets(D_0)
    if isinstance(D_1, utils.Normal_Dataset):
        x_out, y_out = D_1.data_tensor, D_1.target_tensor
    else:
        x_out, y_out = get_data_targets(D_1)
    return x_in, y_in, x_out, y_out


###########################################################
# 审计方法集成化
###########################################################
class LowerBound:
    def __init__(self, D_0, D_1, num_classes, model, T, audit_function):
        self.audit_func = audit_function
        self.eps_OPT: float = .0
        self.eps_LB: float = .0
        self.inference_accuary: float = .0
        self.poisoning_effect: float = .0

        self._epslb(D_0, D_1, num_classes, model, T)

    def _epslb(self, D_0, D_1, num_classes, model, T):
        if self.audit_func == 0:
            self.EPS_LB = EPS_LB_SmipleMI(D_0, D_1, num_classes, model, T)
            self.inference_accuary = self.EPS_LB.inference_acc
        elif self.audit_func == 1:
            self.EPS_LB = EPS_LB_SHADOWMI(D_0, D_1, num_classes, model, T)
            self.inference_accuary = self.EPS_LB.inference_acc
        elif self.audit_func ==2:
            self.EPS_LB = EPS_LB_Memorization(D_0,num_classes)
            self.inference_accuary = self.EPS_LB.inference_acc
        # elif self.audit_func == 2:
        # self.poisoning_Effect = self.EPS_LB.poisoning_effect

        self.eps_OPT = self.EPS_LB.eps_OPT
        self.eps_LB = self.EPS_LB.eps_LB


###########################################################
# 利用基于平均loss的BaseMI， 计算epsilon的下界
###########################################################
class EPS_LB_SmipleMI:
    def __init__(self, D_0, D_1, num_classes, model, T):
        self.D_0=D_0
        self.D_1 = D_1
        self.model = model
        self.T = T
        self._eps_LB()

    def _eps_LB(self):
        x_in,y_in,x_out,y_out = get_X_y(self.D_0,self.D_1)
        count_sum = len(y_in) + len(y_out)
        self.eps_OPT = eps_MI(count_sum, count_sum)

        count = self.T.MI(x_in, y_in) + (len(y_out) - self.T.MI(x_out, y_out))
        self.eps_LB = eps_MI(count, count_sum)
        self.inference_acc = format(float(count) / float(count_sum), '.4f')


###########################################################
# 利用 Shadow_MI 计算epsilon的下界
###########################################################
class EPS_LB_SHADOWMI:
    def __init__(self, D_0, D_1, num_classes, model, T):
        self.D_0=D_0
        self.D_1 = D_1
        self.model = model
        self.T = T
        self._eps_LB()

    def _MI_in_train(self, y, pr):
        res_in = self.T.predict(pr.cpu(), y.cpu(), batch=True)
        count = np.sum(np.argmax(res_in, axis=1))
        return count

    def _MI_out_train(self, y, pr):
        return len(y) - self._MI_in_train(y, pr)

    def _MI(self, y_in, pr_in, y_out, pr_out):
        res_in = self.T.predict(pr_in.cpu(), y_in.cpu(), batch=True)
        res_out = self.T.predict(pr_out.cpu(), y_out.cpu(), batch=True)
        guess_1 = np.argmax(res_in, axis=1)  # 1:in 0:out
        guess_2 = np.argmax(res_out, axis=1)
        # 统计弃权次数,即均判断为D_1
        abstain = np.where((guess_1 + guess_2) == 0)[0].size
        count_sum = len(y_in) - abstain
        # 统计成功次数,即(x,y)判断为D_0,(x,y')判断为D_1
        count = np.where((guess_1 == 1) & (guess_2 == 0))[0].size
        # 统计比较后成功次数,即(x,y)(x,y')均判断为D_0,但(x,y)可能性更大
        compare_indx = np.where((guess_1 == 1) & (guess_2 == 1))[0]
        compare_1 = res_in[:, 1][compare_indx]
        compare_2 = res_out[:, 1][compare_indx]
        count = count + np.where(compare_1 >= compare_2)[0].size
        return count, count_sum

    def _eps_LB(self):
        # 计算最佳统计结果下的隐私损失ε_OPT
        x_in,y_in,x_out,y_out = get_X_y(self.D_0,self.D_1)

        # count_sum = len(self.D_0) + len(self.D_1)
        # 根据模型获取置信度
        predict = predict_proba(x_in, self.model)
        # 基于影子模型隐私推理
        # count = self._MI_in_train(y_in, predict_in) + self._MI_out_train(y_out, predict_out)
        count, count_sum = self._MI(y_in, predict, y_out, predict)
        # 计算ε_LB
        self.eps_OPT = eps_MI(len(y_in), len(y_in))
        self.eps_LB = eps_MI(count, count_sum)
        self.inference_acc = format(float(count) / float(count_sum), '.4f')

# class EPS_LB_SHADOWMI:
#     def __init__(self, D_0, num_classes, model, T):
#         self.D_0 = D_0
#         self.num_classes =num_classes
#         self.model = model
#         self.T = T
#         self._eps_LB()
#
#     def _MI_in_train(self, y, pr):
#         res_in = self.T.predict(pr.cpu(), y.cpu(), batch=True)
#         count = np.sum(np.argmax(res_in, axis=1))
#         return count
#
#     def _eps_LB(self):
#         x_in, y_in = get_data_targets(self.D_0)
#         # x_out, y_out = get_data_targets(self.D_1)
#
#         # 计算最佳统计结果下的隐私损失ε_OPT
#         self.eps_OPT = eps_MI(5000, 5000)
#
#         # count_sum = len(self.D_0) * self.num_classes
#
#         # 根据模型获取置信度
#         predict_in = predict_proba(x_in, self.model)
#         # predict_out = predict_proba(x_out, self.model)
#         # 基于影子模型隐私推理
#         count = self._MI_in_train(y_in, predict_in)
#
#         y_out = torch.stack([torch.tensor(list(set(range(self.num_classes)) - {y}))
#                  for (y) in zip(y_in)]) # 获取除{y}外的元素 Shape:[N,C-1]
#
#         predict_out = []
#         for i in range(self.num_classes-1):
#             res_in = self.T.predict(y_in, predict_in)
#             res_out = self.T.predict
#
#         # 计算ε_LB
#         self.eps_LB = eps_MI(count, count_sum)
#         self.inference_acc = format(float(count) / float(count_sum), '.4f')

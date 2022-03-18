import numpy
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
    def __init__(self, D_0, D_1, num_classes, model, T, args):
        self.eps_OPT: float = .0
        self.eps_LB: float = .0
        self.inference_accuary: float = .0
        self.poisoning_effect: float = .0

        self._epslb(D_0, D_1, num_classes, model, T)

    def _epslb(self, D_0, D_1, num_classes, model, T):
        if self.audit_function == 0:  # 基于loss error的membership inference 的审计办法
            self.EPS_LB = EPS_LB_SmipleMI(D_0, D_1, num_classes, model, T)
            self.inference_accuary = self.EPS_LB.inference_acc
        elif self.audit_function == 1:  # 基于memorization attack的审计方法
            self.EPS_LB = EPS_LB_Memorization(D_0, D_1, num_classes, model)
        elif self.audit_function == 2:  # 基于Shadow model的membership inference 的审计办法
            self.EPS_LB = EPS_LB_SHADOWMI(D_0, D_1, num_classes, model, T)
            self.inference_accuary = self.EPS_LB.inference_acc
        else : #基于Poisoning Attack的审计办法
            if self.binary_classifier == 0:
                self.EPS_LB = EPS_LB_SmipleMI(D_0, D_1, num_classes, model, T)
                self.inference_accuary = self.EPS_LB.inference_acc
            elif self.binary_classifier == 1:
                self.EPS_LB = EPS_LB_Memorization(D_0, D_1, num_classes, model)
            elif self.binary_classifier == 2:
                self.EPS_LB = EPS_LB_SHADOWMI(D_0, D_1, num_classes, model, T)
                self.inference_accuary = self.EPS_LB.inference_acc

        self.eps_OPT = self.EPS_LB.eps_OPT
        self.eps_LB = self.EPS_LB.eps_LB


###########################################################
# 利用基于平均loss的BaseMI， 计算epsilon的下界
###########################################################
class EPS_LB_SmipleMI:
    def __init__(self, D_0, D_1, num_classes, model, T):
        self.D_0 = D_0
        self.D_1 = D_1
        self.model = model
        self.T = T
        self._eps_LB()

    def _eps_LB(self):
        x_in, y_in, x_out, y_out = get_X_y(self.D_0, self.D_1)
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
        self.D_0 = D_0
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
        x_in, y_in, x_out, y_out = get_X_y(self.D_0, self.D_1)

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


###########################################################
# 利用 Memorization Attack 计算epsilon的下界
###########################################################
class EPS_LB_Memorization:
    def __init__(self, D_0, D_1, num_classes, model):
        self.D_0 = D_0
        self.D_1 = D_1
        self.num_classes = num_classes
        self.model = model
        self._eps_LB()

    def _get_confidence(self, predictions: numpy.ndarray, canary_labels):
        canary_labels = np.asarray(canary_labels.cpu())

        c1 = predictions[np.arange(len(canary_labels)), canary_labels]
        c1 = c1.reshape(-1, 1)

        true_labels = np.asarray(self.D_1.target_tensor.cpu())

        incorrect_labels = [  # 获取除{y1,y2}外的元素
            sorted(list(set(range(self.num_classes)) - {y1, y2}))
            for (y1, y2) in zip(true_labels, canary_labels)
        ]
        incorrect_labels = np.asarray(incorrect_labels)

        c2 = []

        for i in range(incorrect_labels.shape[1]):  # 获取|C|-2的incorrect_labels的概率
            c2.append(predictions[np.arange(len(canary_labels)), incorrect_labels[:, i]])
        c2 = np.stack(c2, axis=-1)

        for i in range(len(canary_labels)):
            assert true_labels[i] not in incorrect_labels[i], i
            assert canary_labels[i] not in incorrect_labels[i], i

        return c1, c2

    def eps(self, acc):
        """
        Point estimate of epsilon-DP given the adversary's guessing accuracy
        """
        if acc <= 0.5:
            return 0
        if acc == 1:
            return np.inf
        return np.log(acc / (1 - acc))

    def compute_eps(self, c1, c2, verbose=False):
        # 攻击者猜想： 给定一对labels(y',y'')，攻击者认为更高置信度的label会是canary
        guesses = c1 > c2

        epsilons_low = []
        epsilons_high = []
        N = len(c1)

        # threshold on the max confidence of the two labels: if neither label has high enough confidence, abstain
        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]

        for threshold in thresholds:

            # 当置信度太低时放弃猜测
            dont_know = np.maximum(c1, c2) < threshold

            # 每个canary的猜测次数
            weights = np.sum(1 - dont_know.astype(np.float32), axis=-1)

            # 每个canary的猜测准确率
            accs = np.sum(guesses * (1 - dont_know.astype(np.float32)), axis=-1)
            accs /= np.maximum(
                np.sum((1 - dont_know.astype(np.float32)), axis=-1), 1
            )

            # 只考虑至少进行了一次猜想的canary
            accs = accs[weights > 0]
            weights = weights[weights > 0]
            weights /= np.sum(weights)
            N_effective = len(accs)

            if N_effective:
                # weighted mean 以猜测的频率作为权重，进行加权
                acc_mean = np.sum(accs * weights)

                # weighted std: https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Reliability_weights
                # 加权算数平均的方差
                V = (
                        np.sum(weights * (accs - acc_mean) ** 2)
                        * 1
                        / (1 - np.sum(weights ** 2) + 1e-8)
                )
                acc_std = np.sqrt(V / N_effective)

                # 如果它是错误的方式，反转猜测（DP 定义：是对称的、）
                acc_mean = max(acc_mean, 1 - acc_mean)

                # normal CI：95%双尾置信区间1.96
                acc_low = acc_mean - 1.96 * acc_std
                acc_high = acc_mean + 1.96 * acc_std

                # correction
                acc_low = max(0.5, acc_low)
                acc_high = min(1, acc_high)

                # if all the guesses are correct, treat the accuracy as a Binomial with empirical probability of 1.0
                if acc_mean == 1:
                    # only apply this correction at large enough  sample sizes, else discard as a fluke
                    if N_effective > 50:
                        acc_low = min(acc_low, 1 - 3 / N_effective)
                    else:
                        acc_low = 0.5
            else:
                acc_low, acc_mean, acc_high = 0.0, 0.5, 1.0

            # epsilon CI
            e_low, e_high = self.eps(acc_low), self.eps(acc_high)
            epsilons_low.append(e_low)
            epsilons_high.append(e_high)
            print(
                f"\t{threshold}, DK={np.mean(dont_know):.4f}, N={int(N_effective)}, acc={acc_mean:.4f}, eps=({e_low:.2f}, {e_high:.2f})")

        # 加入考虑了多个阈值，选择最好的一个
        # 与多假设检验校正结合
        eps_low, eps_high = 0, 0
        best_threshold = None

        for el, eh, t in zip(epsilons_low, epsilons_high, thresholds):
            if el > eps_low:
                eps_low = el
                eps_high = eh
                best_threshold = t
            elif (el == eps_low) & (eh > eps_high) & (eh != np.inf):
                eps_high = eh
                best_threshold = t
        return eps_low, eps_high

    def _eps_LB(self):
        if isinstance(self.D_0, utils.Normal_Dataset):
            x_in, y_in = self.D_0.data_tensor, self.D_0.target_tensor
        else:
            x_in, y_in = get_data_targets(self.D_0)

        predictions = predict_proba(x_in, self.model)

        c1, c2 = self._get_confidence(predictions.cpu().numpy(), y_in)

        eps_low, eps_high = self.compute_eps(c1, c2)
        self.eps_OPT = eps_MI(len(y_in), len(y_in))
        self.eps_LB = max(eps_low, eps_high)


###########################################################
# 利用 Shadow_MI |C|-1 个 D_1 计算epsilon的下界
###########################################################
class EPS_LB_SHADOWMI_Multi:
    def __init__(self, D_0, D_1, num_classes, model, T):
        self.D_0 = D_0
        self.D_1 = D_1
        self.num_classes = num_classes
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
        total_count, total_count_sum = 0, 0
        for D_1_i in self.D_1:
            x_in, y_in, x_out, y_out = get_X_y(self.D_0, D_1_i)

            # count_sum = len(self.D_0) + len(self.D_1)
            # 根据模型获取置信度
            predict_in = predict_proba(x_in, self.model)
            predict_out = predict_proba(x_out, self.model)

            # 基于影子模型隐私推理
            count, count_sum = self._MI(y_in, predict_in, y_out, predict_out)
            total_count = total_count + count
            total_count_sum = total_count_sum + count_sum
        # 计算ε_LB
        print("count= ", total_count, "count_sum =", total_count_sum)
        self.eps_OPT = eps_MI(len(y_in) * (self.num_classes - 1), len(y_in) * (self.num_classes - 1))
        self.eps_LB = eps_MI(total_count, total_count_sum)
        self.inference_acc = format(float(total_count) / float(total_count_sum), '.4f')

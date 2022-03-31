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
def eps_MI(count, T, epsilon_theory=0):
    acc_low, acc_high = proportion_confint(count=count, nobs=T, alpha=.05, method="beta")
    print(
        f"count:{count}",
        f"count_sum:{T}",
        f"acc_low: {acc_low}",
        f"acc_high:{acc_high}"
    )
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
    if epsilon_theory != 0 and max(eps_low, eps_high) > epsilon_theory:
        return min(eps_low, eps_high)
    else:
        return max(eps_low, eps_high)


def get_y_in_out(D_0, D_1):
    if isinstance(D_0, utils.Normal_Dataset):
        y_in = D_0.target_tensor
    elif isinstance(D_0, Subset):
        y_in = torch.tensor(np.asarray(D_0.dataset.targets)[D_0.indices])
    else:
        y_in = D_0.targets
        if isinstance(y_in, list):
            y_in = torch.tensor(y_in)
    if isinstance(D_1, utils.Normal_Dataset):
        y_out = D_1.target_tensor
    elif isinstance(D_1, Subset):
        y_out = torch.tensor(np.asarray(D_1.dataset.targets)[D_1.indices])
    else:
        y_out = D_1.targets
        if isinstance(y_out, list):
            y_out = torch.tensor(y_out)
    return y_in, y_out


###########################################################
# 审计方法集成化
###########################################################
class LowerBound:
    def __init__(self, D_0, D_1, num_classes, model, T, args, epsilon_theory):
        self.eps_OPT: float = .0
        self.eps_LB: float = .0
        self.inference_accuary: float = .0
        self.epsilon_theory = epsilon_theory

        self._epslb(D_0, D_1, num_classes, model, T, args)

    def _epslb(self, D_0, D_1, num_classes, model, T, args):
        if args.binary_classifier == 0:  # 基于loss error的membership inference 的审计办法
            self.EPS_LB = EPS_LB_SmipleMI(args.dataset.lower(), D_0, D_1, num_classes, model, T, self.epsilon_theory)
            self.inference_accuary = self.EPS_LB.inference_acc
        elif args.binary_classifier == 1:  # 基于memorization attack的审计方法
            self.EPS_LB = EPS_LB_Memorization(args.dataset.lower(), D_0, D_1, num_classes, model, self.epsilon_theory)
        elif args.binary_classifier == 2:  # 基于Shadow model的membership inference 的审计办法
            self.EPS_LB = EPS_LB_SHADOWMI(args.dataset.lower(), D_0, D_1, num_classes, model, T, self.epsilon_theory)
            self.inference_accuary = self.EPS_LB.inference_acc

        self.eps_OPT = self.EPS_LB.eps_OPT
        self.eps_LB = self.EPS_LB.eps_LB


###########################################################
# 利用基于平均loss的BaseMI， 计算epsilon的下界
###########################################################
class EPS_LB_SmipleMI:
    def __init__(self, dataname, D_0, D_1, num_classes, model, T, epsilon_theory):
        self.dataname = dataname
        self.D_0 = D_0
        self.D_1 = D_1
        self.model = model
        self.T = T
        self._eps_LB(epsilon_theory)

    def _eps_LB(self, epsilon_theory):
        y_in, y_out = get_y_in_out(self.D_0, self.D_1)
        count_sum = len(y_in) + len(y_out)
        self.eps_OPT = eps_MI(count_sum, count_sum)

        count = self.T.MI(self.D_0, y_in) + (len(y_out) - self.T.MI(self.D_1, y_out))
        self.eps_LB = eps_MI(count, count_sum, epsilon_theory)
        self.inference_acc = format(float(count) / float(count_sum), '.4f')


###########################################################
# 利用 Shadow_MI 计算epsilon的下界
###########################################################
class EPS_LB_SHADOWMI:
    def __init__(self, dataname, D_0, D_1, num_classes, model, T, epsilon_theory):
        self.dataname = dataname
        self.D_0 = D_0
        self.D_1 = D_1
        self.model = model
        self.T = T
        self._eps_LB(epsilon_theory)

    # def _MI_in_train(self, y, pr):
    #     res_in = self.T.predict(pr.cpu(), y.cpu(), batch=True)
    #     count = np.sum(np.argmax(res_in, axis=1))
    #     return count
    #
    # def _MI_out_train(self, y, pr):
    #     return len(y) - self._MI_in_train(y, pr)

    def _MI(self, y_in, pr_in, y_out, pr_out):
        res_in = self.T.predict(pr_in.cpu(), y_in.cpu(), batch=True)
        res_out = self.T.predict(pr_out.cpu(), y_out.cpu(), batch=True)
        if np.sum(np.where(res_in[:, 0] == 1.)) == len(y_in):
            count = np.where(np.argmax(res_out, axis=1) == 0)[0].size + len(y_in)
            return count, 2 * len(y_in)
        else:
            guess_1 = np.argmax(res_in, axis=1)  # 0:out 1:in
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

    def _eps_LB(self, epsilon_theory):
        # 计算最佳统计结果下的隐私损失ε_OPT
        y_in, y_out = get_y_in_out(self.D_0, self.D_1)

        # 根据模型获取置信度
        predict = predict_proba(self.D_0, self.model)
        # 基于影子模型隐私推理
        count, count_sum = self._MI(y_in, predict, y_out, predict)
        # 计算ε_LB
        self.eps_OPT = eps_MI(len(y_in), len(y_in))
        self.eps_LB = eps_MI(count, count_sum, epsilon_theory)
        self.inference_acc = format(float(count) / float(count_sum), '.4f')


###########################################################
# 利用 Memorization Attack 计算epsilon的下界
###########################################################
class EPS_LB_Memorization:
    def __init__(self, dataname, D_0, D_1, num_classes, model, epsilon_theory):
        self.dataname = dataname
        self.D_0 = D_0
        self.D_1 = D_1
        self.num_classes = num_classes
        self.model = model
        self._eps_LB(epsilon_theory)

    def _get_y(self,datasset):
        if isinstance(datasset, utils.Normal_Dataset):
            y = datasset.target_tensor
        elif isinstance(datasset, Subset):
            y = torch.tensor(np.asarray(datasset.dataset.targets)[datasset.indices])
        else:
            y = datasset.targets
            if isinstance(y, list):
                y = torch.tensor(y)

        return y

    def _get_confidence(self, predictions: numpy.ndarray, canary_labels):
        canary_labels = np.asarray(canary_labels)

        c1 = predictions[np.arange(len(canary_labels)), canary_labels]
        c1 = c1.reshape(-1, 1)

        true_labels = np.asarray(self._get_y(self.D_1))

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

    def _eps_LB(self, epsilon_theory):
        y_in, y_out = get_y_in_out(self.D_0, self.D_1)

        predictions = predict_proba(self.D_0, self.model)

        c1, c2 = self._get_confidence(predictions.cpu().numpy(), np.asarray(y_in))

        eps_low, eps_high = self.compute_eps(c1, c2)
        self.eps_OPT = eps_MI(len(y_in), len(y_in))
        self.eps_LB = max(eps_low, eps_high)

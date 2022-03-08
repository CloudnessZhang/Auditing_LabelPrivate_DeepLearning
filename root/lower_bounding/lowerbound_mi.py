import numpy as np
from torch import nn
from torch.utils.data import Subset

from utils import clopper_pearson, get_data_targets, predict_proba
from binary_classifier.inference.attack_model import AttackModels
from binary_classifier.inference import base_MI


# 根据统计结果计算隐私损失
def eps_MI(count, T):
    acc_low, acc_high = clopper_pearson(count, T, .05)
    acc_low = max(acc_low, 1 - acc_low)
    acc_high = max(acc_high, 1 - acc_high)
    # 计算ε_LB
    if acc_low == 0.5 or acc_low == 0.5:
        return 0
    elif acc_low == 1 or acc_high == 1:
        return np.inf
    else:
        eps_low = np.log(acc_low / (1 - acc_low))
        eps_high = np.log(acc_high / (1 - acc_high))
    return max(eps_low, eps_high)


###########################################################
# 审计方法集成化
###########################################################
class LowerBound:
    def __init__(self, D_trn, D_tst, model, T, audit_function):
        self.audit_func = audit_function
        self.eps_OPT :float =.0
        self.eps_LB :float =.0
        self.inference_Accuary :float =.0
        self.poisoning_Effect :float =.0

        self._epslb(D_trn, D_tst, model, T)

    def _epslb(self, D_trn, D_tst, model, T):
        if self.audit_func == 0:
            self.EPS_LB = EPS_LB_SmipleMI(D_trn, D_tst, model, T)
            self.inference_Accuary = self.EPS_LB.inference_acc
        elif self.audit_func == 1:
            self.EPS_LB = EPS_LB_SHADOWMI(D_trn, D_tst, model, T)
        # elif self.audit_func == 2:
        #     self.poisoning_Effect = self.EPS_LB.poisoning_effect

        self.eps_OPT= self.EPS_LB.eps_OPT
        self.eps_LB = self.EPS_LB.eps_LB


###########################################################
# 利用基于平均loss的BaseMI， 计算epsilon的下界
###########################################################
class EPS_LB_SmipleMI:
    def __init__(self, D_trn, D_tst, model, T):
        self.D_trn = D_trn
        self.D_tst = D_tst
        self.model = model
        self.T = T
        self._eps_LB()

    def eps_LB(self):
        count_sum = len(self.D_trn) + len(self.D_tst)
        self.eps_OPT = eps_MI(count_sum, count_sum)

        count = self.T.MI(self.D_trn) + (len(self.D_tst) - self.T.MI(self.D_tst))
        self.eps_LB = eps_MI(count, count_sum)
        self.inference_acc = format(float(count) / float(count_sum), '.4f')


###########################################################
# 利用 Shadow_MI 计算epsilon的下界
###########################################################
class EPS_LB_SHADOWMI:
    def __init__(self, D_trn, D_tst, model, T):
        self.D_trn = D_trn
        self.D_tst = D_tst
        self.model = model
        self.T = T
        self._eps_LB()

    def _MI_in_train(self, y, pr):
        res_in = self.T.predict(pr.cpu(), y.cpu(), batch=True)
        count = np.sum(np.argmax(res_in, axis=1))
        return count

    def _MI_out_train(self, y, pr):
        return len(y) - self._MI_in_train(y, pr)

    def _eps_LB(self):
        trn_x, trn_y = get_data_targets(self.D_trn)
        tst_x, tst_y = get_data_targets(self.D_tst)

        # 计算最佳统计结果下的隐私损失ε_OPT
        count_sum = len(self.D_trn) + len(self.D_tst)
        self.eps_OPT = eps_MI(count_sum, count_sum)

        # 根据模型获取置信度
        predict_train = predict_proba(trn_x, self.model)
        predict_test = predict_proba(tst_x, self.model)
        # 基于影子模型隐私推理
        count = self._MI_in_train(trn_y, predict_train) + self._MI_out_train(tst_y, predict_test)
        # 计算ε_LB
        self.eps_LB = eps_MI(count, count_sum)
        self.inference_acc = format(float(count) / float(count_sum), '.4f')


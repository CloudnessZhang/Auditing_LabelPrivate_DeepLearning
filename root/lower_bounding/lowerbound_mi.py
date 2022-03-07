import numpy as np
from torch import nn
from torch.utils.data import Subset

from utils import clopper_pearson,get_data_targets,predict_proba
from binary_classifier.inference.attack_model import AttackModels
from binary_classifier.inference import base_MI


def MI_in_train(y, predict, T):
    res_in = T.predict(predict, y, batch=True)
    return np.sum(res_in)


def MI_out_train(y, predict, T):
    return len(y) - MI_in_train(y, predict, T)


def eps_MI(count, T):
    acc_low, acc_high = clopper_pearson(count, T, .05)
    acc_low = max(acc_low, 1 - acc_low)
    acc_high = max(acc_high, 1 - acc_high)
    # 计算ε_LB
    if acc_low == 0.5 or acc_low == 0.5:
        return 0
    elif acc_low == 1 or acc_high == 1:
        return 1
    else:
        eps_low = np.log(acc_low / (1 - acc_low))
        eps_high = np.log(acc_high / (1 - acc_high))
    return max(eps_low, eps_high)


###########################################################
# 利用 Shadow_MI 计算epsilon的下界
###########################################################
def eps_LB_Shadow(D_trn, D_tst, model, T):
    trn_x, trn_y = get_data_targets(D_trn)
    tst_x, tst_y = get_data_targets(D_tst)
    # 根据模型获取置信度
    predict_train = predict_proba(trn_x,model)
    predict_test = predict_proba(tst_x,model)
    # 基于影子模型隐私推理
    count = MI_in_train(trn_y, predict_train, T) + MI_out_train(tst_y, predict_test, T)
    # 计算ε_LB
    eps1 = eps_MI(count, len(D_trn) + len(D_tst))

    ##############################################
    # D_train,D_test 大小相同
    #################################################
    D_trn_new = Subset(D_trn, range(0, len(D_tst)))
    trn_x_new, trn_y_new = get_data_targets(D_trn_new)
    predict_trn_new = predict_proba(trn_x_new,model)
    # 基于影子模型隐私推理
    count_new = MI_in_train(trn_y_new, predict_trn_new, T) + MI_out_train(tst_y, predict_test, T)
    # 计算ε_LB
    eps2 = eps_MI(count_new, len(D_trn_new) + len(D_tst))
    return eps1, eps2


def eps_LB_BaseMI(D_trn, D_tst, model, T):
        count1 = T.MI(D_trn)+(len(D_tst)-T.MI(D_tst))
        eps3 = eps_MI(count1, len(D_trn) + len(D_tst))
        ##############################################
        # D_train,D_test 大小相同
        #################################################
        D_trn_new = Subset(D_trn, range(0, len(D_tst)))
        T_new = base_MI(D_trn_new, model)
        count2 = T_new.MI(D_trn)+(len(D_tst)-T.MI(D_tst))
        eps4 = eps_MI(count1, len(D_trn) + len(D_tst))
        return eps3, eps4

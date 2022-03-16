from math import sqrt

from torch import nn
from sklearn.metrics import mean_squared_error

import utils
from utils import get_data_targets, predict_proba,predict


class BaseMI:
    def __init__(
            self,
            D_train=None,  # 训练模型的数据集
            net=None
    ) -> None:
        self.D_train = D_train
        self.net = net
        if (net is not None) and (D_train is not None):
          self.threshould = self._get_mean()

    def _get_mean(self):
        trn_x, trn_y = get_data_targets(self.D_train)
        ptrdict_y=predict(trn_x, self.net)
        mse = mean_squared_error(trn_y.cpu(),ptrdict_y.cpu())
        return sqrt(mse)

    def MI(self, x, y):
        predict_y = predict(x, self.net)
        errors = y-predict_y
        count = (abs(errors) <= self.threshould).sum()
        return count.cpu()

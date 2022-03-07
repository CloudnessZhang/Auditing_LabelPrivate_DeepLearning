from torch import nn

from utils import get_data_targets, predict_proba


class BaseMI:
    def __init__(
            self,
            D_train,  # 训练模型的数据集
            net
    ) -> None:
        self.D_train = D_train
        self.net = net
        self.loss_mean = self._get_mean()

    def _get_mean(self):
        trn_x, trn_y = get_data_targets(self.D_train)
        loss = nn.CrossEntropyLoss()
        predict_trn = predict_proba(trn_x, self.net)

        trn_loss = loss(predict_trn, trn_y)
        trn_loss_mean = trn_loss.mean()
        return trn_loss_mean

    def MI(self, D):
        threshould = self.loss_mean
        x, y = get_data_targets(D)
        predict = predict_proba(x, self.net)
        count = (predict >= threshould).sum()
        return count

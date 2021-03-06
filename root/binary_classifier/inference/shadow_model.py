from typing import List, Tuple, Dict
from copy import copy

import torch
from tqdm import tqdm_notebook
import numpy as np

import utils
from utils import Normal_Dataset, get_data_targets, predict_proba


#
# try:
#     import tensorflow as tf
# except (ModuleNotFoundError, ImportError):
#     import warnings
#
#     warnings.warn("Tensorflow is not installed")


class ShadowModels:
    """
    Creates a swarm of shadow models and trains them with a split
    of the synthetic data.

    Parameters
    ----------
    X: ndarray or DataFrame

    y: ndarray or str
        if X it's a DataFrame then y must be the target column name,
        otherwise

    n_models: int
        number of shadow models to build. Higher number returns
        better results but is limited by the number of records
        in the input data.

    target_classes: int
        number of classes of the target model or lenght of the
        prediction array of the target model.

    learner: learner? #fix type
        learner to use as shadow model. It must be as similar as
        possible to the target model. It must have `predict_proba`
        method. Now only sklearn learners are implemented.

    Returns
    -------

    ShadowModels object
    """

    def __init__(
            self,
            dataname,
            train_set,
            test_set,
            n_models: int,
            target_classes: int,
            learner,
            **fit_kwargs,
    ) -> None:
        self.dataname =dataname
        self.transform = dataname.upper() + '_TRAIN_TRANS'
        self.n_models = n_models
        self.train_set = train_set
        self.test_set = test_set
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # if self.X_train.ndim > 1:
        #     # flatten images or matrices inside 1rst axis
        #     self.X_train = self.X_train.reshape(self.X_train.shape[0], -1)
        #     self.X_test = self.X_test.reshape(self.X_test.shape[0], -1)

        # self.y_train = np.array(y_train)
        # self.y_test = np.array(y_test)
        self.target_classes = target_classes
        self.train_splits = self._split_data(self.dataname, self.train_set, self.n_models, self.target_classes)
        self.test_splits = self._split_data(self.dataname, self.test_set, self.n_models, self.target_classes)
        self.learner = learner
        self.models = self._make_model_list(self.learner, self.n_models)

        # train models
        self.results = self.train_predict_shadows(**fit_kwargs)

    @staticmethod
    def _split_data(dataname, data_set, n_splits: int, n_classes: int) -> List[np.ndarray]:
        """
        Split manually into n datasets maintaining class proportions
        """
        if isinstance(data_set, utils.Normal_Dataset):
            data, targets = data_set.data_tensor, data_set.target_tensor
        else:
            data, targets = get_data_targets(data_set,dataname)

        classes = range(n_classes)
        class_partitions_X = []
        class_partitions_y = []
        # Split by class
        for clss in classes:
            X_clss = data[targets == clss]
            y_clss = targets[targets == clss]
            batch_size = len(X_clss) // n_splits
            splits_X = []
            splits_y = []
            splits = []
            for i in range(n_splits):
                split_X = X_clss[i * batch_size: (i + 1) * batch_size, :]
                split_y = y_clss[i * batch_size: (i + 1) * batch_size]
                splits_X.append(split_X.cpu())
                splits_y.append(split_y.cpu())
            class_partitions_X.append(splits_X)
            class_partitions_y.append(splits_y)

        # -------------------
        # consolidate splits into ndarrays
        # -------------------

        grouped_X = []
        grouped_y = []
        for split in range(n_splits):
            parts_X = []
            parts_y = []
            for part_X, part_y in zip(class_partitions_X, class_partitions_y):
                parts_X.append(part_X[split])
                parts_y.append(part_y[split])
            grouped_X.append(parts_X)
            grouped_y.append(parts_y)

        splits_X = []
        splits_y = []
        for group_X, group_y in zip(grouped_X, grouped_y):
            splits_X.append(np.vstack(group_X))
            splits_y.append(np.hstack(group_y))

        return (splits_X, splits_y)

    @staticmethod
    def _make_model_list(learner, n) -> List:
        """
        Intances n shadow models, copies of the input parameter learner
        """
        # try:
        #     if isinstance(learner, tf.keras.models.Model):
        #         models = [copy(learner) for _ in range(n)]
        # except NameError:
        #     print("using sklearn shadow models")
        #     pass
        #
        # if isinstance(learner, BaseEstimator):
        #     models = [clone(learner) for _ in range(n)]

        models = [copy(learner) for _ in range(n)]

        return models

    def train_predict_shadows(self, **fit_kwargs):
        """
        "in" : 1
        "out" : 0
        """

        # TRAIN and predict
        results = []
        for i, (model, X_train, y_train, X_test, y_test) in enumerate(tqdm_notebook(
                zip(self.models, self.train_splits[0], self.train_splits[1], self.test_splits[0], self.test_splits[1]))):
            print("??????shadow model_"+ str(i))
            shadow_train_set = Normal_Dataset((torch.tensor(X_train), torch.tensor(y_train)),self.dataname,self.transform)
            net = model.train(shadow_train_set)
            # data IN training set labeled 1

            predict_in = predict_proba(shadow_train_set, net)  # predict(data), (n_samples, n_classes)
            y_train = y_train.reshape(-1, 1)
            res_in = np.hstack((predict_in.cpu(), y_train, np.ones_like(y_train)))

            # data OUT of training set, labeled 0
            shadow_test_set = Normal_Dataset((torch.tensor(X_test),torch.tensor(y_test)),self.dataname,self.transform)
            predict_out = predict_proba(shadow_test_set, net)
            y_test = y_test.reshape(-1, 1)
            res_out = np.hstack((predict_out.cpu(), y_test, np.zeros_like(y_test)))

            # concat in single array
            model_results = np.vstack((res_in, res_out))
            results.append(model_results)

        results = np.vstack(results)
        return results

    def __repr__(self):
        rep = (
            f"Shadow models: {self.n_models}, {self.learner.__class__}\n"
            f"lengths of data splits : {[len(s) for s in self._splits]}"
        )
        return rep

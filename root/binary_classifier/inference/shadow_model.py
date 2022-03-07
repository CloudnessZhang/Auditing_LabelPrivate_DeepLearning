from typing import List, Tuple, Dict
from copy import copy

import torch
from torch.utils.data import Subset
from tqdm import tqdm_notebook
import torch.utils.data as data
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.base import clone, BaseEstimator

from utils import Normal_Dataset,get_data_targets

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
            train_set: Subset,
            test_set: Subset,
            n_models: int,
            target_classes: int,
            learner,
            **fit_kwargs,
    ) -> None:

        self.n_models = n_models
        self.train_set = train_set
        self.test_set = test_set
        # if self.X_train.ndim > 1:
        #     # flatten images or matrices inside 1rst axis
        #     self.X_train = self.X_train.reshape(self.X_train.shape[0], -1)
        #     self.X_test = self.X_test.reshape(self.X_test.shape[0], -1)

        # self.y_train = np.array(y_train)
        # self.y_test = np.array(y_test)
        self.target_classes = target_classes
        self.train_splits = self._split_data(self.train_set, self.n_models, self.target_classes)
        self.test_splits = self._split_data(self.test_set, self.n_models, self.target_classes)
        self.learner = learner
        self.models = self._make_model_list(self.learner, self.n_models)

        # train models
        self.results = self.train_predict_shadows(**fit_kwargs)

    @staticmethod
    def _split_data(
            data_set: Subset, n_splits: int, n_classes: int
    ) -> List[np.ndarray]:
        """
        Split manually into n datasets maintaining class proportions
        """
        data, targets = get_data_targets(data_set)

        classes = range(n_classes)
        class_partitions_X = []
        class_partitions_y = []
        # Split by class
        for clss in classes:
            # targets = [cifar.dataset.targets[i] for i in cifar.indices]
            inds = np.where(np.array(targets) == clss)
            X_clss = np.array(data)[inds]
            y_clss = np.array(targets)[inds]
            X_clss = torch.stack(X_clss.tolist(), 0)
            batch_size = len(X_clss) // n_splits
            splits_X = []
            splits_y = []
            for i in range(n_splits):
                split_X = X_clss[i * batch_size: (i + 1) * batch_size, :]
                split_y = y_clss[i * batch_size: (i + 1) * batch_size]
                splits_X.append(split_X)
                splits_y.append(split_y)
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
        for model, X_train, y_train, X_test, y_test in tqdm_notebook(
                zip(self.models, self.train_splits[0], self.train_splits[1], self.test_splits[0], self.test_splits[1])):
            model.set_trainset(Normal_Dataset((X_train, y_train)))
            # model.train_model() #暂时

            # data IN training set labeled 1
            y_train = y_train.reshape(-1, 1)
            predict_in = model.predict_proba(X_train) #predict(data), (n_samples, n_classes)
            res_in = np.hstack((predict_in.detach().numpy(), y_train, np.ones_like(y_train)))

            # data OUT of training set, labeled 0
            y_test = y_test.reshape(-1, 1)
            predict_out = model.predict_proba(X_test)
            res_out = np.hstack((predict_out.detach().numpy(), y_test, np.zeros_like(y_test)))

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

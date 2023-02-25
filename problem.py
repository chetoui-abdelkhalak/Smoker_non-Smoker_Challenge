import os
import datetime

import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss, recall_score, precision_score

from sklearn.model_selection import ShuffleSplit

import rampwf as rw
from rampwf.score_types.base import BaseScoreType
from rampwf.score_types.classifier_base import ClassifierBaseScoreType
from rampwf.workflows.sklearn_pipeline import SKLearnPipeline
from rampwf.workflows.sklearn_pipeline import Estimator

from sklearn.metrics import f1_score
from sklearn.metrics import log_loss

problem_title = "Smoker-non Smoker classification"


# -----------------------------------------------------------------------------
# Worklow element
# -----------------------------------------------------------------------------



# -----------------------------------------------------------------------------
# Predictions type
# -----------------------------------------------------------------------------


_prediction_label_names = [0, 1]

Predictions = rw.prediction_types.make_multiclass(label_names=_prediction_label_names)

workflow = rw.workflows.Estimator()



# -----------------------------------------------------------------------------
# Score types
# -----------------------------------------------------------------------------

'''
class PointwiseLogLoss(BaseScoreType):
    # subclass BaseScoreType to use raw y_pred (proba's)
    is_lower_the_better = True
    minimum = 0.0
    maximum = np.inf

    def __init__(self, name="pw_ll", precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        score = log_loss(y_true[:, 1:], y_pred[:, 1:])
        return score


class PointwisePrecision(ClassifierBaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name="pw_prec", precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true_label_index, y_pred_label_index):
        score = precision_score(y_true_label_index, y_pred_label_index)
        return score


class PointwiseRecall(ClassifierBaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name="pw_rec", precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true_label_index, y_pred_label_index):
        score = recall_score(y_true_label_index, y_pred_label_index)
        return score

score_types = [
    # log-loss
    PointwiseLogLoss(),
    # point-wise (for each time step) precision and recall
    PointwisePrecision(),
    PointwiseRecall(),
]

class F1(BaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name="f1-score", precision=4):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        f1 = f1_score(y_true, y_pred)
        return f1

score_types = [
    F1(name="f1-score"),
]
'''
score_types = [
    rw.score_types.ROCAUC(name='auc'),
    rw.score_types.Accuracy(name='acc'),
    rw.score_types.NegativeLogLikelihood(name='nll'),
]

# -----------------------------------------------------------------------------
# Cross-validation scheme
# -----------------------------------------------------------------------------


'''
def get_cv(X, y):
    cv = ShuffleSplit(n_splits=8, test_size=0.20, random_state=42)
    return cv.split(X, y)
'''
def get_cv(X, y):
    # using 5 folds as default
    k = 5
    # up to 10 fold cross-validation based on 5 splits, using two parts for
    # testing in each fold
    n_splits = 5
    cv = KFold(n_splits=n_splits)
    splits = list(cv.split(X, y))
    # 5 folds, each point is in test set 4x
    # set k to a lower number if you want less folds
    pattern = [
        ([2, 3, 4], [0, 1]),
        ([0, 1, 4], [2, 3]),
        ([0, 2, 3], [1, 4]),
        ([0, 1, 3], [2, 4]),
        ([1, 2, 4], [0, 3]),
        ([0, 1, 2], [3, 4]),
        ([0, 2, 4], [1, 3]),
        ([1, 2, 3], [0, 4]),
        ([0, 3, 4], [1, 2]),
        ([1, 3, 4], [0, 2]),
    ]
    for ps in pattern[:k]:
        yield (
            np.hstack([splits[p][1] for p in ps[0]]),
            np.hstack([splits[p][1] for p in ps[1]]),
        )

# -----------------------------------------------------------------------------
# Training / testing data reader
# -----------------------------------------------------------------------------


def _read_data(path, type_):
    data = pd.read_csv(os.path.join(path, 'data','data_'+ type_))
    #data = data.set_index('ID')
    data = data.drop(columns=['ID'],axis=1)
    data=data.drop(columns=['oral'],axis=1)
    data['dental caries']=data['dental caries'].astype('int')
    data['dental caries']=data['dental caries'].astype('str')

    y = pd.read_csv(os.path.join(path, 'data','labels_'+ type_))
    y = y.drop(columns=['ID'],axis=1)
    y = y['smoking']

    return data, y


def get_train_data(path="."):
    return _read_data(path, "train.csv")


def get_test_data(path="."):
    return _read_data(path, "test.csv")

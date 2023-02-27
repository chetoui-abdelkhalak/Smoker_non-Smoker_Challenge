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

workflow = rw.workflows.Estimator()


# -----------------------------------------------------------------------------
# Predictions type
# -----------------------------------------------------------------------------


_prediction_label_names = [0, 1]

Predictions = rw.prediction_types.make_multiclass(label_names=_prediction_label_names)




# -----------------------------------------------------------------------------
# Score types
# -----------------------------------------------------------------------------


score_types = [
    rw.score_types.ROCAUC(name='auc'),
    rw.score_types.Accuracy(name='acc'),
    rw.score_types.NegativeLogLikelihood(name='nll'),
]

# -----------------------------------------------------------------------------
# Cross-validation scheme
# -----------------------------------------------------------------------------


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

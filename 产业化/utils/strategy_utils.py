# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     train_utils
   Description:   ...
   Author:        cqh
   date:          2022/7/28 22:14
-------------------------------------------------
   Change Activity:
                  2022/7/28:
-------------------------------------------------
"""
__author__ = 'cqh'

from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.calibration import CalibratedClassifierCV

random_state = 2022


def smote_process(data_x, data_y):
    """已经标准化了的数据"""
    # SMOTE
    pipeline = Pipeline([('over', SMOTE(random_state=random_state, sampling_strategy={0: 16359, 1: 4641 * 4}))])
    all_data_x, all_data_y = pipeline.fit_resample(data_x, data_y)

    return all_data_x, all_data_y


def random_under_process(data_x, data_y):
    pipeline = Pipeline([('under', RandomUnderSampler(random_state=random_state))])
    all_data_x, all_data_y = pipeline.fit_resample(data_x, data_y)
    return all_data_x, all_data_y


def random_over_process(data_x, data_y):
    pipeline = Pipeline([('over', RandomOverSampler(random_state=random_state))])
    all_data_x, all_data_y = pipeline.fit_resample(data_x, data_y)
    return all_data_x, all_data_y


def train_strategy(clf, strategy, X_train, y_train, X_test):
    """
    不同策略进行训练
    :param X_test:
    :param y_train:
    :param X_train:
    :param clf:
    :param strategy:
                1 : 直接fit
                2 : fit后校准
                3 : 校准后fits
    :return:
    """
    if strategy == 1:
        pass
    elif strategy == 2:
        clf.fit(X_train, y_train)
        clf = CalibratedClassifierCV(clf, cv="prefit")
    elif strategy == 3:
        clf = CalibratedClassifierCV(clf, cv=5)
    else:
        raise ValueError("Not found strategy!")
    clf.fit(X_train, y_train)
    train_prob = clf.predict_proba(X_train)[:, 1]
    test_prob = clf.predict_proba(X_test)[:, 1]
    return train_prob, test_prob

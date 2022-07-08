# -*- coding: gbk -*-
"""
-------------------------------------------------
   File Name:     sub_sample_train
   Description:   ...
   Author:        cqh
   date:          2022/7/5 10:13
-------------------------------------------------
   Change Activity:
                  2022/7/5:
-------------------------------------------------
"""
__author__ = 'cqh'

import os
import warnings

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier

from my_logger import MyLog
from utils_api import get_train_test_data

warnings.filterwarnings('ignore')

def get_local_xgb_para(xgb_thread_num=-1, xgb_boost_num=50):
    """personal xgb para"""
    params = {
        'booster': 'gbtree',
        'max_depth': 11,
        'min_child_weight': 7,
        'subsample': 1,
        'colsample_bytree': 0.7,
        'eta': 0.05,
        'objective': 'binary:logistic',
        'nthread': xgb_thread_num,
        'verbosity': 0,
        'eval_metric': 'logloss',
        'seed': 2022,
        'tree_method': 'hist'
    }
    num_boost_round = xgb_boost_num
    return params, num_boost_round


def multi_score(model, all_samples):
    all_samples_y = all_samples['Label']
    all_samples_x = all_samples.drop(['ID', 'Label'], axis=1)

    auc = cross_val_score(model, all_samples_x, all_samples_y, scoring='roc_auc', cv=5)
    my_logger.info(f"AUC: {auc.mean()} \n")
    return auc.mean()


def model_train(fit_data_x, fit_data_y, test_data_x, test_data_y):

    # LR
    lr_auc = lr_train(fit_data_x, fit_data_y, test_data_x, test_data_y)

    # XGB
    xgb_auc = xgb_train(fit_data_x, fit_data_y, test_data_x, test_data_y)

    return [lr_auc, xgb_auc]


def lr_train(fit_data_x, fit_data_y, test_data_x, test_data_y):
    lr_local = LogisticRegression(solver="liblinear", n_jobs=1, max_iter=max_iter)
    lr_local.fit(fit_data_x, fit_data_y)
    y_predict = lr_local.predict_proba(test_data_x)[:, 1]
    lr_auc = roc_auc_score(test_data_y, y_predict)
    return lr_auc


def xgb_train(fit_data_x, fit_data_y, test_data_x, test_data_y):
    d_train_local = xgb.DMatrix(fit_data_x, label=fit_data_y)
    params, num_boost_round = get_local_xgb_para()
    xgb_local = xgb.train(params=params,
                          dtrain=d_train_local,
                          num_boost_round=num_boost_round,
                          verbose_eval=False,
                          xgb_model=None)
    d_test_local = xgb.DMatrix(test_data_x)
    predict_prob = xgb_local.predict(d_test_local)
    xgb_auc = roc_auc_score(test_data_y, predict_prob)
    return xgb_auc


if __name__ == '__main__':

    my_logger = MyLog().logger

    max_iter = 100
    # get data
    train_data, test_data = get_train_test_data()

    train_data_y = train_data['Label']
    train_data_x = train_data.drop(['ID', 'Label'], axis=1)
    test_data_y = test_data['Label']
    test_data_x = test_data.drop(['Label', 'ID'], axis=1)

    # 分割断
    frac_list = np.arange(0.7, 1.01, 0.05)

    all_auc_list = []
    for frac in frac_list:
        my_logger.warning(f"================== {int(frac * 100)} % ===================")
        fit_data = train_data.sample(frac=frac)
        fit_data_y = fit_data['Label']
        fit_data_x = fit_data.drop(['ID', 'Label'], axis=1)

        auc_list = model_train(fit_data_x, fit_data_y, test_data_x, test_data_y)
        print("auc_list[lr_auc, xgb_auc]: ", auc_list)
        all_auc_list.append(auc_list)

    result_df = pd.DataFrame(data=all_auc_list, index=frac_list, columns=['lr_auc', 'xgb_auc'])
    result_df.to_csv(f"./result/S01_frac_all_auc_v2.csv")

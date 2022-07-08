# -*- coding: gbk -*-
"""
-------------------------------------------------
   File Name:     pca_test
   Description:   ...
   Author:        cqh
   date:          2022/7/5 21:04
-------------------------------------------------
   Change Activity:
                  2022/7/5:
-------------------------------------------------
"""
__author__ = 'cqh'

import time

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
import pandas as pd
import xgboost as xgb
import numpy as np

from utils_api import get_train_test_data, covert_time_format


def get_local_xgb_para(xgb_thread_num=1, xgb_boost_num=50):
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
        'seed': 998,
        'tree_method': 'hist'
    }
    num_boost_round = xgb_boost_num
    return params, num_boost_round


def xgb_train(fit_train_x=None, fit_train_y=None, fit_test_x=None, fit_test_y=None):
    d_train_local = xgb.DMatrix(fit_train_x, label=fit_train_y)
    params, num_boost_round = get_local_xgb_para()
    xgb_local = xgb.train(params=params,
                          dtrain=d_train_local,
                          num_boost_round=num_boost_round,
                          verbose_eval=False,
                          xgb_model=None)
    d_test_local = xgb.DMatrix(fit_test_x)
    predict_prob = xgb_local.predict(d_test_local)
    score = roc_auc_score(fit_test_y, predict_prob)
    return score


def pca_params_comp():
    """
    比较不同components数值对应不同的消耗时间
    :return:
    """
    max_len = test_data_x.shape[1]
    # components_list = [i for i in range(3000, max_len, 100)]
    components_list = np.arange(0.95, 1.01, 0.05)

    fit_time_list = []
    tra_time_list = []
    var_ratio_list = []
    print("start...")

    for components in components_list:
        start_time = time.time()
        pca = PCA(n_components=components)
        new_test_data = pca.fit_transform(test_data_x)
        fit_tra_time = time.time()
        pca.transform(train_data_x)
        tra_time = time.time()

        fit_cost_time = covert_time_format(fit_tra_time - start_time)
        tra_cost_time = covert_time_format(tra_time - fit_tra_time)
        var_ratio = pca.explained_variance_ratio_.sum()

        # info show
        print("pca.n_components_:", pca.n_components_)
        print("pca.svd_solver:", pca.svd_solver)
        print("pca.explained_variance_ratio_.sum():", var_ratio)
        print("shape: ", test_data_x.shape, new_test_data.shape)
        print("fit_time: ", fit_cost_time)
        print("tra_time: ", tra_cost_time)
        print("==========================================")

        # build list
        fit_time_list.append(fit_cost_time)
        tra_time_list.append(tra_cost_time)
        var_ratio_list.append(var_ratio)

    result_dict = {
        "n_components": components_list,
        "fit_time": fit_time_list,
        "tra_time": tra_time_list,
        "var_ratio": var_ratio_list
    }

    result_df = pd.DataFrame(data=result_dict)
    result_df.to_csv("./result/pca_auc_comp.csv")

    print("done!")


if __name__ == '__main__':
    train_data, test_data = get_train_test_data()
    test_data_x = test_data.drop(['Label', 'ID'], axis=1)
    train_data_x = train_data.drop(['Label', 'ID'], axis=1)
    print("get data...")
    pca_params_comp()

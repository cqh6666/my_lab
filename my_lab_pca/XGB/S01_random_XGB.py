# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     S01_random_XGB
   Description:   ...
   Author:        cqh
   date:          2022/7/12 10:08
-------------------------------------------------
   Change Activity:
                  2022/7/12:
-------------------------------------------------
"""
__author__ = 'cqh'

import sys

import xgboost as xgb
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd

from utils_api import get_train_test_data
from xgb_utils_api import get_local_xgb_para, get_xgb_model_pkl
from my_logger import MyLog

def xgb_train(select_rate=0.1, xgb_model=None, local_boost=50):

    fit_train = train_data.sample(frac=select_rate)
    # 处理test_data
    fit_train_y = fit_train['Label']
    fit_train_x = fit_train.drop(['Label'], axis=1)

    d_train_local = xgb.DMatrix(fit_train_x, label=fit_train_y)

    xgb_local = xgb.train(params=params,
                          dtrain=d_train_local,
                          num_boost_round=local_boost,
                          verbose_eval=False,
                          xgb_model=xgb_model)
    d_test_local = xgb.DMatrix(test_data_x)
    test_y_predict = xgb_local.predict(d_test_local)
    auc = roc_auc_score(test_data_y, test_y_predict)

    return auc


def sub_all_train(select_rate=0.1, xgb_model=None, local_boost=50):
    auc_list = []
    for i in range(run_num):
        auc = xgb_train(select_rate=select_rate, xgb_model=xgb_model, local_boost=local_boost)
        auc_list.append(auc)
    return np.mean(auc_list)


def range_train():
    num_boost_list = [i for i in range(50, 501, 50)]
    range_list = np.arange(0.05, 1.01, 0.05)

    tra_res_dict = {}
    no_tra_res_dict = {}

    for iter_idx in num_boost_list:
        select_tra_res = []
        select_no_tra_res = []
        for select in range_list:
            tra_auc = sub_all_train(select, xgb_model, iter_idx)
            no_tra_auc = sub_all_train(select, None, iter_idx)
            select_tra_res.append(tra_auc)
            select_no_tra_res.append(no_tra_auc)

        tra_res_dict[f'tra_{iter_idx}'] = select_tra_res
        no_tra_res_dict[f'no_tra_{iter_idx}'] = select_no_tra_res

        pd.DataFrame(tra_res_dict, index=range_list).to_csv(f"./result/tra_xgb_auc_comp_{iter_idx}.csv")
        pd.DataFrame(no_tra_res_dict, index=range_list).to_csv(f"./result/no_tra_xgb_auc_comp_{iter_idx}.csv")
        my_logger.info("save success!")


if __name__ == '__main__':

    my_logger = MyLog().logger

    run_num = 20
    select_ratio = 0.1
    xgb_thread_num = -1
    xgb_boost_num = 50
    is_transfer = sys.argv[1]
    params, num_boost_round = get_local_xgb_para(xgb_thread_num=xgb_thread_num, num_boost_round=xgb_boost_num)
    xgb_model = get_xgb_model_pkl(is_transfer)

    # 获取数据
    train_data, test_data = get_train_test_data()
    # 处理数据
    train_data.set_index(["ID"], inplace=True)
    test_data.set_index(["ID"], inplace=True)

    # 处理test_data
    test_data_y = test_data['Label']
    test_data_x = test_data.drop(['Label'], axis=1)

    range_train()


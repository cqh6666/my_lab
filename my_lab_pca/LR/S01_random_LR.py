# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     S01_random_LR
   Description:   ...
   Author:        cqh
   date:          2022/7/11 20:57
-------------------------------------------------
   Change Activity:
                  2022/7/11:
-------------------------------------------------
"""
__author__ = 'cqh'

import sys
import time
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed

from lr_utils_api import get_transfer_weight
from my_logger import MyLog
from utils_api import get_train_test_data, covert_time_format
import pandas as pd
warnings.filterwarnings('ignore')


def get_random_train_data(select_rate):
    sub_train_data = train_data.sample(frac=select_rate)
    # 处理train_data
    train_data_y = sub_train_data['Label']
    train_data_x = sub_train_data.drop(['Label'], axis=1)
    return train_data_x, train_data_y


def sub_global_train(select_rate=0.1, is_transfer=1, local_iter=100):
    """
    选取10%的数据进行训练
    :param select_rate:
    :param is_transfer:
    :param local_iter:
    :return:
    """
    fit_train_x, fit_train_y = get_random_train_data(select_rate)

    if is_transfer == 1:
        fit_train_x = fit_train_x * global_feature_weight
        fit_test_x = test_data_x * global_feature_weight
    else:
        fit_train_x = fit_train_x
        fit_test_x = test_data_x

    lr_local = LogisticRegression(max_iter=local_iter, solver="liblinear")
    lr_local.fit(fit_train_x, fit_train_y)
    y_predict = lr_local.decision_function(fit_test_x)
    auc = roc_auc_score(test_data_y, y_predict)

    # my_logger.info(
    #     f'[sub_global] - solver:liblinear, transfer:{is_transfer}, max_iter:{local_iter}, train_iter:{lr_local.n_iter_}, auc: {auc}')
    return auc


def range_sub_train(iter_idx, is_tra=1):
    range_list = np.arange(0.05, 1.01, 0.05)

    tra_res_dict = {}
    select_res = []
    for select in range_list:
        if is_tra == 1:
            cur_auc = run(select, 1, iter_idx)
        else:
            cur_auc = run(select, 0, iter_idx)
        select_res.append(cur_auc)

    tra_res_dict[f'tra_{is_tra}_{iter_idx}'] = select_res
    pd.DataFrame(tra_res_dict, index=range_list).to_csv(f"./result/S01_tra_{is_tra}_lr_auc_comp_{iter_idx}.csv")
    my_logger.info("save success!")


def run(select_rate, is_transfer, local_iter):
    s_t = time.time()
    # 匹配相似样本（从训练集） XGB建模 多线程
    auc_list = []
    for i in range(run_round):
        auc = sub_global_train(select_rate, is_transfer, local_iter)
        auc_list.append(auc)

    mean_auc = np.mean(auc_list)
    e_t = time.time()
    my_logger.info(f"select_rate:{select_rate}, is_transfer:{is_transfer}, local_iter:{local_iter}, cost_time:{covert_time_format(e_t-s_t)}, auc:{mean_auc}")
    return np.mean(auc_list)


if __name__ == '__main__':

    my_logger = MyLog().logger

    select_rate = 0.1
    is_transfer = int(sys.argv[2])
    local_iter = int(sys.argv[1])

    pool_nums = 20
    run_round = 10

    train_data, test_data = get_train_test_data()
    # 处理数据
    train_data.set_index(["ID"], inplace=True)
    test_data.set_index(["ID"], inplace=True)

    # 处理test_data
    test_data_y = test_data['Label']
    test_data_x = test_data.drop(['Label'], axis=1)

    global_feature_weight = get_transfer_weight(is_transfer)

    # transfer: select_rate, auc_25, auc_50, auc_100, auc_150, auc_200, auc_250 ...]

    range_sub_train(local_iter, is_transfer)


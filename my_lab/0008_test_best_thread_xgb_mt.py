# -*- coding: gbk -*-
"""
-------------------------------------------------
   File Name:     0008_test_best_thread_xgb_mt
   Description:   ...
   Author:        cqh
   date:          2022/5/17 15:20
-------------------------------------------------
   Change Activity:
                  2022/5/17:
-------------------------------------------------
"""
__author__ = 'cqh'

import threading
import time
import numpy as np
import pandas as pd
from random import shuffle
import xgboost as xgb
import warnings
import os
from my_logger import MyLog
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED, as_completed
import sys
from gc import collect
import pickle


def get_local_xgb_para(xgb_thread_num):
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


def xgb_train(xgb_thread_num):
    # train_start_time = time.time()
    d_train_local = xgb.DMatrix(train_x, label=train_y)
    params, num_boost_round = get_local_xgb_para(xgb_thread_num)

    xgb.train(params=params,
              dtrain=d_train_local,
              num_boost_round=num_boost_round,
              verbose_eval=False)
    # train_end_time = time.time()
    # run_time = round(train_end_time - train_start_time, 2)
    # my_logger.info(f"xgb_thread_num: {xgb_thread_num} | cost: {run_time} s")


if __name__ == '__main__':
    DATA_SOURCE_PATH = f"/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/data/24h/"  # 训练集的X和Y

    # 训练集的X和Y
    train_x = pd.read_feather(
        os.path.join(DATA_SOURCE_PATH, "all_x_test_24h_norm_dataframe_999_miss_medpx_max2dist.feather"))
    train_y = pd.read_feather(
        os.path.join(DATA_SOURCE_PATH, "all_y_test_24h_norm_dataframe_999_miss_medpx_max2dist.feather"))['Label']

    pool_nums = 15
    n_thread = 10
    xgb_boost_num = 70
    n_iter = 200
    my_logger = MyLog().logger
    my_logger.warning(f"[params] - xgb_boost_num:{xgb_boost_num}, n_thread:{n_thread}, pool_nums:{pool_nums}, n_iter:{n_iter}")

    run_time_list = []
    n_thread_list = [1, 2, 3, 4]
    pool_nums_list = [15, 20, 25, 30]
    for p_n in pool_nums_list:
        cur_run_time = []
        for n_t in n_thread_list:
            thread_start_time = time.time()
            with ThreadPoolExecutor(max_workers=p_n) as executor:
                thread_list = []
                for i in range(n_iter):
                    thread = executor.submit(xgb_train, n_t)
                    thread_list.append(thread)

                wait(thread_list, return_when=ALL_COMPLETED)
            thread_run_time = round(time.time() - thread_start_time, 2)
            cur_run_time.append(thread_run_time)
            my_logger.info(f"pool_nums_idx: {p_n} | n_thread_idx: {n_t} | cost:{thread_run_time} s \n")
            collect()
        run_time_list.append(cur_run_time)

    my_logger.warning("all thread cost time list:")
    my_logger.info(run_time_list)

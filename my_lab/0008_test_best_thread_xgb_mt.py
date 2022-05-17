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
    train_start_time = time.time()
    d_train_local = xgb.DMatrix(train_x, label=train_y)
    params, num_boost_round = get_local_xgb_para(xgb_thread_num)

    xgb.train(params=params,
              dtrain=d_train_local,
              num_boost_round=num_boost_round,
              verbose_eval=False)
    train_end_time = time.time()
    run_time = round(train_end_time - train_start_time, 2)
    my_logger.info(f"xgb_thread_num: {xgb_thread_num} | cost: {run_time} s")
    return run_time


if __name__ == '__main__':
    DATA_SOURCE_PATH = f"/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/data/24h/"  # 训练集的X和Y

    # 训练集的X和Y
    train_x = pd.read_feather(
        os.path.join(DATA_SOURCE_PATH, "all_x_test_24h_norm_dataframe_999_miss_medpx_max2dist.feather"))
    train_y = pd.read_feather(
        os.path.join(DATA_SOURCE_PATH, "all_y_test_24h_norm_dataframe_999_miss_medpx_max2dist.feather"))['Label']

    pool_nums = 25
    n_thread = 10
    xgb_boost_num = 70
    n_iter = 200
    my_logger = MyLog().logger
    all_avg_time = []
    my_logger.warning(f"[params] - xgb_boost_num:{xgb_boost_num}, n_thread:{n_thread}, pool_nums:{pool_nums}, n_iter:{n_iter}")
    for n_t in range(1, n_thread):
        avg_time = []
        thread_start_time = time.time()
        with ThreadPoolExecutor(max_workers=pool_nums) as executor:
            thread_list = []
            for i in range(n_iter):
                thread = executor.submit(xgb_train, n_t)
                thread_list.append(thread)

            for future in as_completed(thread_list):
                avg_time.append(thread.result())
        cur_avg_time = np.mean(avg_time)
        all_avg_time.append(cur_avg_time)
        thread_end_time = time.time()
        my_logger.info(f"n_thread_idx: {n_t} | build {n_iter} all cost:{thread_end_time - thread_start_time} s | avg cost:{cur_avg_time} s \n")
        collect()

    my_logger.warning("all thread cost time list:")
    my_logger.info(all_avg_time)

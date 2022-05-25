# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     0006_global_LR
   Description:   全局数据的LR得到weight
   Author:        cqh
   date:          2022/5/23 10:39
-------------------------------------------------
   Change Activity:
                  2022/5/23:
-------------------------------------------------
"""
__author__ = 'cqh'

import warnings
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
from multiprocessing import Pool

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from my_logger import MyLog
import time
import os
import pandas as pd

warnings.filterwarnings('ignore')


def global_train(idx):

    start_time = time.time()

    lr_all = LogisticRegression(solver='liblinear')

    lr_all.fit(x_train, y_train)

    # feature weight
    # weight_importance = lr_all.coef_[0]

    # predict
    y_predict = lr_all.decision_function(x_test)
    auc = roc_auc_score(y_test, y_predict)

    run_time = round(time.time() - start_time, 2)
    time.sleep(1)
    my_logger.info(f'[{idx}] train cost time: {run_time} s, auc: {auc}')


if __name__ == '__main__':
    pre_hour = 24
    pool_nums = 5

    DATA_SOURCE_PATH = f"/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/data/{pre_hour}h/"

    my_logger = MyLog().logger

    breast_cancer = load_breast_cancer()
    data_x = breast_cancer.data
    data_y = breast_cancer.target
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=45)

    start_time = time.time()
    # # 多进程
    # pool = Pool(processes=pool_nums)
    # for iter_idx in range(0, 100, 1):
    #     pool.apply_async(func=global_train)
    # pool.close()
    # pool.join()

    my_logger.warning("start mt training...")
    # # 多线程
    with ThreadPoolExecutor(max_workers=pool_nums) as executor:
        thread_list = []
        for i in range(0, 100, 1):
            thread = executor.submit(global_train, i)
            thread_list.append(thread)
        wait(thread_list, return_when=ALL_COMPLETED)


    run_time = round(time.time() - start_time, 2)

    my_logger.info(f"train all models time: {run_time} s")

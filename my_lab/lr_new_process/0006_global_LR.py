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

import pickle
import random
import sys
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from my_logger import MyLog
import time
import os
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def get_train_test_data():
    train_x = pd.read_feather(
        os.path.join(DATA_SOURCE_PATH, "all_x_train_24h_norm_dataframe_999_miss_medpx_max2dist.feather"))
    train_y = pd.read_feather(
        os.path.join(DATA_SOURCE_PATH, "all_y_train_24h_norm_dataframe_999_miss_medpx_max2dist.feather"))['Label']
    test_x = pd.read_feather(
        os.path.join(DATA_SOURCE_PATH, "all_x_test_24h_norm_dataframe_999_miss_medpx_max2dist.feather"))
    test_y = pd.read_feather(
        os.path.join(DATA_SOURCE_PATH, "all_y_test_24h_norm_dataframe_999_miss_medpx_max2dist.feather"))['Label']

    return train_x, train_y, test_x, test_y


def save_weight_importance_to_csv(weight_important, max_iter):
    weight_importance = [abs(i) for i in weight_important]
    weight_importance = [i / sum(weight_importance) for i in weight_importance]
    weight_importance_df = pd.DataFrame({"init_weight": weight_importance})
    wi_file_name = os.path.join(MODEL_SAVE_PATH, f"0006_{pre_hour}h_global_lr_liblinear_{max_iter}.csv")
    weight_importance_df.to_csv(wi_file_name, index=False)
    my_logger.info(f"save to csv success! - {wi_file_name}")


def global_train(max_iter):

    start_time = time.time()

    train_x_ft = train_x
    test_x_ft = test_x

    lr_all = LogisticRegression(max_iter=max_iter, n_jobs=1, solver="liblinear")
    lr_all.fit(train_x_ft, train_y)
    y_predict = lr_all.decision_function(test_x_ft)
    auc = roc_auc_score(test_y, y_predict)

    run_time = round(time.time() - start_time, 2)

    # save feature weight
    weight_importance = lr_all.coef_[0]
    save_weight_importance_to_csv(weight_importance, max_iter)
    # save model
    model_file_name = os.path.join(MODEL_SAVE_PATH, f"{pre_hour}h_global_lr_liblinear_{max_iter}.pkl")
    pickle.dump(lr_all, open(model_file_name, "wb"))
    my_logger.info(f"save lr model to pkl - [{model_file_name}]")

    my_logger.info(f'[global] - solver:liblinear, max_iter:{max_iter}, train_iter:{lr_all.n_iter_}, cost time: {run_time} s, auc: {auc}')


def get_train_data_for_random_idx(select_rate=0.1):
    """
    选取10%的索引列表，利用这索引列表随机选取数据
    :param train_x:
    :param train_y:
    :param select_rate:
    :return:
    """
    len_split = int(train_x.shape[0] * select_rate)
    random_idx = random.sample(list(range(train_x.shape[0])), len_split)

    train_x_ = train_x.loc[random_idx, :]
    train_x_.reset_index(drop=True, inplace=True)

    train_y_ = train_y.loc[random_idx]
    train_y_.reset_index(drop=True, inplace=True)

    return train_x_, train_y_


def sub_global_train(max_iter):
    """
    选取10%的数据进行训练
    :param train_x:
    :param train_y:
    :return:
    """
    start_time = time.time()
    train_x, train_y = get_train_data_for_random_idx(select_rate=0.1)
    train_x_ft = train_x
    test_x_ft = test_x

    lr_all = LogisticRegression(max_iter=max_iter, n_jobs=1, solver="liblinear")
    lr_all.fit(train_x_ft, train_y)
    y_predict = lr_all.decision_function(test_x_ft)
    auc = roc_auc_score(test_y, y_predict)

    run_time = round(time.time() - start_time, 2)

    my_logger.info(f'[global] - solver:liblinear, max_iter:{max_iter}, train_iter:{lr_all.n_iter_}, cost time: {run_time} s, auc: {auc}')


if __name__ == '__main__':
    # lr_max_iter = int(sys.argv[1])
    pre_hour = 24
    pool_nums = 20
    root_dir = f"{pre_hour}h"
    DATA_SOURCE_PATH = f"/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/data/{root_dir}/"
    MODEL_SAVE_PATH = f'/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/result/personal_model_with_lr/{root_dir}/global_model/'
    my_logger = MyLog().logger

    train_x, train_y, test_x, test_y = get_train_test_data()

    max_iter = int(sys.argv[1])
    # select_rate = 0.1
    # train_x, train_y = get_train_data_for_random_idx(train_x, train_y, select_rate)
    # del train_x, train_y

    # start_time = time.time()
    # pool = Pool(processes=pool_nums)
    # for iter_idx in range(100, 3000, 100):
    #     pool.apply_async(func=global_train, args=(train_x, train_y, test_x, test_y, iter_idx))
    # pool.close()
    # pool.join()
    # my_logger.warning(f"lr_max_iter:{lr_max_iter}")
    sub_global_train(max_iter)
    # my_logger.warning("start mt training...")
    # # 多线程
    # with ThreadPoolExecutor(max_workers=pool_nums) as executor:
    #     thread_list = []
    #     for idx in range(1000):
    #         thread = executor.submit(global_train, 100)
    #         thread_list.append(thread)
    #     wait(thread_list, return_when=ALL_COMPLETED)
    #
    # run_time = round(time.time() - start_time, 2)
    # my_logger.info(f"build all models need: {run_time} s")
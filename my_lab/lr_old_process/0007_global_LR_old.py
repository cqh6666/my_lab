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

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from my_logger import MyLog
import time
import os
import pandas as pd
import warnings

warnings.filterwarnings('ignore')


def get_train_test_data():
    train_x = pd.read_feather(
        os.path.join(DATA_SOURCE_PATH, "all_x_train_24_df_rm1_norm1.feather"))
    train_y = pd.read_feather(
        os.path.join(DATA_SOURCE_PATH, "all_y_train_24_df_rm1_norm1.feather"))['Label']
    test_x = pd.read_feather(
        os.path.join(DATA_SOURCE_PATH, "all_x_test_24_df_rm1_norm1.feather"))
    test_y = pd.read_feather(
        os.path.join(DATA_SOURCE_PATH, "all_y_test_24_df_rm1_norm1.feather"))['Label']

    return train_x, train_y, test_x, test_y


def save_weight_importance_to_csv(weight_important, max_iter):
    # 不标准化 初始特征重要性
    weight_importance_df = pd.DataFrame({"feature_weight": weight_important})
    weight_importance_df.to_csv(transfer_weight_file, index=False)
    my_logger.info(f"save to csv success! - {transfer_weight_file}")

    # 标准化 用作psm_0
    weight_importance = [abs(i) for i in weight_important]
    normalize_weight_importance = [i / sum(weight_importance) for i in weight_importance]
    normalize_weight_importance_df = pd.DataFrame({"normalize_weight": normalize_weight_importance})
    normalize_weight_importance_df.to_csv(init_psm_weight_file, index=False)
    my_logger.info(f"save to csv success! - {init_psm_weight_file}")


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
    pickle.dump(lr_all, open(model_file_name_file, "wb"))
    my_logger.info(f"save lr model to pkl - [{model_file_name_file}]")

    my_logger.info(
        f'[global] - solver:liblinear, max_iter:{max_iter}, train_iter:{lr_all.n_iter_}, cost time: {run_time} s, auc: {auc}')


def get_train_data_for_random_idx(train_x, train_y, select_rate):
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

    my_logger.info(f"select sub train x shape {train_x_.shape}.")
    return train_x_, train_y_


def sub_global_train(select_rate=0.1, is_transfer=1, local_iter=100):
    """
    选取10%的数据进行训练
    :param select_rate:
    :param is_transfer:
    :param local_iter:
    :return:
    """
    start_time = time.time()

    train_x_ft, train_y_ft = get_train_data_for_random_idx(train_x, train_y, select_rate)

    if is_transfer == 1:
        fit_train_x = train_x_ft * global_feature_weight
        fit_test_x = test_x * global_feature_weight
    else:
        fit_train_x = train_x_ft
        fit_test_x = test_x

    lr_local = LogisticRegression(max_iter=local_iter, solver="liblinear")
    lr_local.fit(fit_train_x, train_y_ft)
    y_predict = lr_local.decision_function(fit_test_x)
    auc = roc_auc_score(test_y, y_predict)

    run_time = round(time.time() - start_time, 2)

    my_logger.info(
        f'[sub_global] - solver:liblinear, max_iter:{local_iter}, train_iter:{lr_local.n_iter_}, cost time: {run_time} s, auc: {auc}')
    return auc


if __name__ == '__main__':
    # lr_max_iter = int(sys.argv[1])
    pre_hour = 24
    pool_nums = 20
    root_dir = f"{pre_hour}h_old2"
    DATA_SOURCE_PATH = f"/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/data/{root_dir}/"
    MODEL_SAVE_PATH = f'/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/result/personal_model_with_lr/{root_dir}/global_model/'

    max_iter = 400
    local_max_iter = int(sys.argv[1])
    select_rate = int(sys.argv[2])
    model_file_name_file = os.path.join(MODEL_SAVE_PATH, f"0007_{pre_hour}h_global_lr_{max_iter}.pkl")
    transfer_weight_file = os.path.join(MODEL_SAVE_PATH, f"0007_{pre_hour}h_global_weight_lr_{max_iter}.csv")
    init_psm_weight_file = os.path.join(MODEL_SAVE_PATH, f"0007_{pre_hour}h_0_psm_global_lr_{max_iter}.csv")

    my_logger = MyLog().logger

    train_x, train_y, test_x, test_y = get_train_test_data()
    # global_train(max_iter)

    global_feature_weight = pd.read_csv(transfer_weight_file).squeeze().tolist()

    range_sub_train()
    # start_time = time.time()
    # pool = Pool(processes=pool_nums)
    # for iter_idx in range(100, 3000, 100):
    #     pool.apply_async(func=global_train, args=(train_x, train_y, test_x, test_y, iter_idx))
    # pool.close()
    # pool.join()
    # my_logger.warning(f"lr_max_iter:{lr_max_iter}")
    # my_logger.warning("start mt training...")
    # 多线程
    # with ThreadPoolExecutor(max_workers=pool_nums) as executor:
    #     thread_list = []
    #     for idx in range(100, 1001, 100):
    #         thread = executor.submit(global_train, idx)
    #         thread_list.append(thread)
    #     wait(thread_list, return_when=ALL_COMPLETED)
    #
    # run_time = round(time.time() - start_time, 2)
    # my_logger.info(f"build all models need: {run_time} s")
    #
    # start_time = time.time()
    # with ThreadPoolExecutor(max_workers=pool_nums) as executor:
    #     thread_list = []
    #     for idx in range(50, 1001, 50):
    #         thread = executor.submit(sub_global_train, idx)
    #         thread_list.append(thread)
    #     wait(thread_list, return_when=ALL_COMPLETED)
    #
    # run_time = round(time.time() - start_time, 2)
    # my_logger.info(f"build all models need: {run_time} s")

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
import sys
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from my_logger import MyLog
import time
import os
import pandas as pd


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


def get_train_test_data_2():
    train_x = pd.read_feather(
        os.path.join(DATA_SOURCE_PATH, "all_x_train_24_df_rm1_miss2_norm2.feather"))
    train_y = pd.read_feather(
        os.path.join(DATA_SOURCE_PATH, "all_y_train_24_df_rm1_miss2_norm2.feather"))['Label']
    test_x = pd.read_feather(
        os.path.join(DATA_SOURCE_PATH, "all_x_test_24_df_rm1_miss2_norm2.feather"))
    test_y = pd.read_feather(
        os.path.join(DATA_SOURCE_PATH, "all_y_test_24_df_rm1_miss2_norm2.feather"))['Label']

    return train_x, train_y, test_x, test_y


def get_train_test_data_3():
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
    weight_importance = [abs(i) for i in weight_important]
    weight_importance = [i / sum(weight_importance) for i in weight_importance]
    weight_importance_df = pd.DataFrame({"init_weight": weight_importance})
    wi_file_name = os.path.join(MODEL_SAVE_PATH, f"0006_{pre_hour}h_global_lr_{max_iter}.csv")
    weight_importance_df.to_csv(wi_file_name, index=False)
    my_logger.info(f"save to csv success! - {wi_file_name}")


def global_train(max_iter):

    start_time = time.time()
    lr_all = LogisticRegression(solver='liblinear', max_iter=max_iter)
    lr_all.fit(train_x, train_y)

    # feature weight
    weight_importance = lr_all.coef_[0]
    save_weight_importance_to_csv(weight_importance, max_iter)

    # predict
    y_predict = lr_all.decision_function(test_x)
    auc = roc_auc_score(test_y, y_predict)

    run_time = round(time.time() - start_time, 2)

    # save model
    model_file_name = os.path.join(MODEL_SAVE_PATH, f"{pre_hour}h_global_lr_{max_iter}.pkl")
    pickle.dump(lr_all, open(model_file_name, "wb"))
    my_logger.info(f"save lr model to pkl - [{model_file_name}]")

    my_logger.info(f'{max_iter} | cost time: {run_time} s, auc: {auc}')


if __name__ == '__main__':
    lr_max_iter = int(sys.argv[1])
    pre_hour = 24
    pool_nums = 25
    # DATA_SOURCE_PATH = f"/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/data/{pre_hour}h/"
    DATA_SOURCE_PATH = f"/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/data/{pre_hour}h_old/"
    MODEL_SAVE_PATH = f'/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/result/personal_model_with_lr/{pre_hour}h_old/global_model/'
    my_logger = MyLog().logger

    train_x, train_y, test_x, test_y = get_train_test_data_3()

    start_time = time.time()
    # pool = Pool(processes=pool_nums)
    # for iter_idx in range(100, 3000, 100):
    #     pool.apply_async(func=global_train, args=(train_x, train_y, test_x, test_y, iter_idx))
    # pool.close()
    # pool.join()
    my_logger.warning(f"lr_max_iter:{lr_max_iter}")
    global_train(lr_max_iter)
    # my_logger.warning("start mt training...")
    # # 多线程
    # with ThreadPoolExecutor(max_workers=pool_nums) as executor:
    #     thread_list = []
    #     for i in range(0, 1000, 1):
    #         thread = executor.submit(global_train, i)
    #         thread_list.append(thread)
    #     wait(thread_list, return_when=ALL_COMPLETED)

    # run_time = round(time.time() - start_time, 2)
    # my_logger.info(f"build all models need: {run_time} s")

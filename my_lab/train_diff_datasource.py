# -*- coding: gbk -*-
"""
-------------------------------------------------
   File Name:get_data_from_feather
   Description:111
   Author:cqh
   date:2022/4/13 11:26
-------------------------------------------------
   Change Activity:
                   2022/4/13:
-------------------------------------------------
"""
__author__ = 'cqh'

import os
import time

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def get_train_test_data():
    """药物距离 处理MED PX 0 miss2_norm2
    """
    load_time = time.time()

    train_x = pd.read_feather(
        os.path.join(DATA_SOURCE_PATH, "all_x_train_24_df_rm1_miss2_norm2.feather"))
    train_y = pd.read_feather(
        os.path.join(DATA_SOURCE_PATH, "all_y_train_24_df_rm1_miss2_norm2.feather"))['Label']
    test_x = pd.read_feather(
        os.path.join(DATA_SOURCE_PATH, "all_x_test_24_df_rm1_miss2_norm2.feather"))
    test_y = pd.read_feather(
        os.path.join(DATA_SOURCE_PATH, "all_y_test_24_df_rm1_miss2_norm2.feather"))['Label']

    print(f"load data needs: {time.time() - load_time} s")

    return train_x, train_y, test_x, test_y


def get_train_test_data2():
    """药物距离 不处理MED PX 0 miss1_norm2
    """
    load_time = time.time()

    train_x = pd.read_feather(
        os.path.join(DATA_SOURCE_PATH, "all_x_train_24_df_rm1_norm2.feather"))
    train_y = pd.read_feather(
        os.path.join(DATA_SOURCE_PATH, "all_y_train_24_df_rm1_norm2.feather"))['Label']
    test_x = pd.read_feather(
        os.path.join(DATA_SOURCE_PATH, "all_x_test_24_df_rm1_norm2.feather"))
    test_y = pd.read_feather(
        os.path.join(DATA_SOURCE_PATH, "all_y_test_24_df_rm1_norm2.feather"))['Label']

    print(f"load data needs: {time.time() - load_time} s")
    return train_x, train_y, test_x, test_y


def get_train_test_data3():
    """
    药物距离 处理MED PX 只对LAB标准化 miss2_norm1
    :return:
    """
    train_x = pd.read_feather(
        os.path.join(DATA_SOURCE_PATH, "all_x_train_24h_norm_dataframe_999_miss_medpx_max2dist.feather"))
    train_y = pd.read_feather(
        os.path.join(DATA_SOURCE_PATH, "all_y_train_24h_norm_dataframe_999_miss_medpx_max2dist.feather"))['Label']
    test_x = pd.read_feather(
        os.path.join(DATA_SOURCE_PATH, "all_x_test_24h_norm_dataframe_999_miss_medpx_max2dist.feather"))
    test_y = pd.read_feather(
        os.path.join(DATA_SOURCE_PATH, "all_y_test_24h_norm_dataframe_999_miss_medpx_max2dist.feather"))['Label']

    return train_x, train_y, test_x, test_y

def get_train_test_data4():
    """yuanborong 48 data 不是用药距离而是用药次数"""
    load_time = time.time()
    train_df = pd.read_csv(os.path.join(YUAN_DATA_SOURCE_PATH, "train-data-no-empty-normalized.csv"))
    test_df = pd.read_csv(os.path.join(YUAN_DATA_SOURCE_PATH, "test-data-no-empty-normalized.csv"))
    train_y = train_df['Label']
    train_x = train_df.drop(['Label'], axis=1)
    test_y = test_df['Label']
    test_x = test_df.drop(['Label'], axis=1)
    print(f"load data needs: {time.time() - load_time} s")
    return train_x, train_y, test_x, test_y


def get_diff_data(data_flag):
    if data_flag == 1:
        name_flag = "miss2_norm2"
        train_x, train_y, test_x, test_y = get_train_test_data()
        return train_x, train_y, test_x, test_y, name_flag
    elif data_flag == 2:
        name_flag = "miss1_norm2"
        train_x, train_y, test_x, test_y = get_train_test_data2()
        return train_x, train_y, test_x, test_y, name_flag
    elif data_flag == 3:
        name_flag = "miss2_norm1"
        train_x, train_y, test_x, test_y = get_train_test_data3()
        return train_x, train_y, test_x, test_y, name_flag
    elif data_flag == 4:
        name_flag = "yuan_lab"
        train_x, train_y, test_x, test_y = get_train_test_data4()
        return train_x, train_y, test_x, test_y, name_flag


def train_diff_data(data_flag):
    """
    进行LR训练并输出预测值和消耗时间
    不同标志代表不同数据集
    :param data_flag:
    :return:
    """
    train_x, train_y, test_x, test_y, flag = get_diff_data(data_flag)
    print(train_x.info)
    start_time = time.time()
    lr = LogisticRegression(solver='liblinear')
    lr.fit(train_x, train_y)
    y_predict = lr.decision_function(test_x)
    auc = roc_auc_score(test_y, y_predict)
    print(f"{flag} - n_iter:{lr.n_iter_}, auc:{auc}, cost time: {time.time() - start_time} s")


if __name__ == '__main__':
    # run()
    DATA_SOURCE_PATH = f"/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/data/24h/"
    YUAN_DATA_SOURCE_PATH = "/panfs/pfs.local/work/liu/xzhang_sta/yuanborong/data/df_data/pre_48h/"

    train_diff_data(4)
    train_diff_data(1)
    train_diff_data(2)
    train_diff_data(3)

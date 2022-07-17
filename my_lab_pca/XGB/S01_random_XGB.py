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

from utils_api import get_train_test_data, save_to_csv_by_row
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
                          xgb_model=xgb_model,
                          verbose_eval=False)

    d_test_local = xgb.DMatrix(test_data_x)
    test_y_predict = xgb_local.predict(d_test_local)
    auc = roc_auc_score(test_data_y, test_y_predict)

    return auc


def sub_all_train(select_rate, xgb_model, local_boost):
    auc_list = []
    for i in range(run_num):
        auc = xgb_train(select_rate=select_rate, xgb_model=xgb_model, local_boost=local_boost)
        auc_list.append(auc)
    return np.mean(auc_list)


def range_train(num_boost, is_tra=1):
    """
    得到抽取 0.05 ~ 1 样本训练的auc性能变化
    :param num_boost:
    :param is_tra:
    :return:
    """
    select_res = []
    for select in select_range_list:
        if is_tra == 1:
            cur_auc = sub_all_train(select, xgb_model, num_boost)
        else:
            cur_auc = sub_all_train(select, None, num_boost)
        select_res.append(cur_auc)

    res_df = pd.DataFrame(data=[select_res], columns=select_range_list, index=[xgb_boost_num])
    save_to_csv_by_row(csv_file_path, res_df)
    my_logger.info(f"save to csv success! - AUC: {select_res}")


if __name__ == '__main__':
    my_logger = MyLog().logger

    xgb_boost_num = int(sys.argv[1])
    is_transfer = int(sys.argv[2])
    print(xgb_boost_num, is_transfer)
    run_num = 20
    select_ratio = 0.1
    xgb_thread_num = -1

    # 保存抽取不同数据时，对应的树的迭代次数的AUC变化情况
    csv_file_path = f"./result/S01_xgb_auc_tra{is_transfer}.csv"

    params, num_boost_round = get_local_xgb_para(xgb_thread_num=xgb_thread_num, num_boost_round=xgb_boost_num)
    xgb_model = get_xgb_model_pkl(is_transfer)

    select_range_list = np.around(np.arange(0.05, 1.01, 0.05), 2)

    # 获取数据
    train_data, test_data = get_train_test_data()
    # 处理数据
    train_data.set_index(["ID"], inplace=True)
    test_data.set_index(["ID"], inplace=True)

    # 处理test_data
    test_data_y = test_data['Label']
    test_data_x = test_data.drop(['Label'], axis=1)

    range_train(xgb_boost_num, is_transfer)

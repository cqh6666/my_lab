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
from sklearn.model_selection import train_test_split
from utils_api import get_train_test_data, get_train_test_x_y
from my_logger import MyLog
import time
import os
import pandas as pd
import warnings

warnings.filterwarnings('ignore')


def save_weight_importance_to_csv(weight_important):
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


def global_train():
    start_time = time.time()

    train_x_ft = train_x
    test_x_ft = test_x

    lr_all = LogisticRegression(max_iter=max_iter, solver="liblinear")
    lr_all.fit(train_x_ft, train_y)
    y_predict = lr_all.decision_function(test_x_ft)
    auc = roc_auc_score(test_y, y_predict)

    run_time = round(time.time() - start_time, 2)

    # save feature weight
    weight_importance = lr_all.coef_[0]
    save_weight_importance_to_csv(weight_importance)

    # save model
    pickle.dump(lr_all, open(model_file_name_file, "wb"))
    my_logger.info(f"save lr model to pkl - [{model_file_name_file}]")

    my_logger.info(
        f'[global] - solver:liblinear, max_iter:{max_iter}, train_iter:{lr_all.n_iter_}, cost time: {run_time} s, auc: {auc}')


if __name__ == '__main__':
    MODEL_SAVE_PATH = f"/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/code_pca/code_lr/result/global_model/"
    pre_hour = 24
    max_iter = 400
    model_file_name_file = os.path.join(MODEL_SAVE_PATH, f"0007_{pre_hour}h_global_lr_{max_iter}.pkl")
    transfer_weight_file = os.path.join(MODEL_SAVE_PATH, f"0007_{pre_hour}h_global_weight_lr_{max_iter}.csv")
    init_psm_weight_file = os.path.join(MODEL_SAVE_PATH, f"0007_{pre_hour}h_0_psm_global_lr_{max_iter}.csv")

    my_logger = MyLog().logger

    train_x, train_y, test_x, test_y = get_train_test_x_y()
    global_train()

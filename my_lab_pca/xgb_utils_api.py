# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     xgb_utils_api
   Description:   获取xgb相关信息，文件
   Author:        cqh
   date:          2022/7/8 14:30
-------------------------------------------------
   Change Activity:
                  2022/7/8:
-------------------------------------------------
"""
__author__ = 'cqh'
import pandas as pd
import os
import pickle

pre_hour = 24
root_dir = f"{pre_hour}h_old2"
MODEL_SAVE_PATH = f'/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/result/psm_with_xgb/{root_dir}/global_model/'


def get_xgb_model_pkl(is_transfer):
    # 迁移模型
    if is_transfer == 1:
        xgb_model_file = os.path.join(MODEL_SAVE_PATH, f"0007_{pre_hour}h_global_xgb_boost500.pkl")
        xgb_model = pickle.load(open(xgb_model_file, "rb"))
        return xgb_model
    else:
        return None


def get_init_similar_weight():
    init_similar_weight_file = os.path.join(MODEL_SAVE_PATH, f'0007_{pre_hour}h_global_xgb_feature_weight_boost500.csv')
    init_similar_weight = pd.read_csv(init_similar_weight_file).squeeze().tolist()
    return init_similar_weight


def get_local_xgb_para(xgb_thread_num=1, num_boost_round=50):
    """
    xgb local 参数
    :param xgb_thread_num: 线程数
    :param num_boost_round: 数迭代次数
    :return:
    """
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
        'seed': 2022,
        'tree_method': 'hist'
    }
    return params, num_boost_round

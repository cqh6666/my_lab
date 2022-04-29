# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     0006_test_xgb_nthread
   Description:   ...
   Author:        cqh
   date:          2022/4/27 19:17
-------------------------------------------------
   Change Activity:
                  2022/4/27:
-------------------------------------------------
"""
__author__ = 'cqh'

import pandas as pd
import xgboost as xgb
import time
from my_logger import MyLog
import os

# 自定义日志
my_logger = MyLog().logger

global_num_boost = 300
transfer_num_boost = 100
key_component = '24h_all_999_normalize'
train_x = pd.read_feather(
    f'/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/data/24h/{key_component}_train_x_data.feather')
train_y = \
pd.read_feather(f'/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/data/24h/{key_component}_train_y_data.feather')[
    'Label']

# 迁移模型保存路径
SAVE_PATH = '/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/result/personal_model_with_xgb/'


def train_global_model(num_boost):
    d_train = xgb.DMatrix(train_x, label=train_y)
    params = {
        'booster': 'gbtree',
        'max_depth': 8,
        'min_child_weight': 7,
        'eta': 0.15,
        'objective': 'binary:logistic',
        'nthread': 20,
        'verbosity': 0,
        'eval_metric': 'logloss',
        'subsample': 0.5,
        'colsample_bytree': 0.5,
        'tree_method': 'hist',
        'seed': 1001,
    }

    start_time = time.time()
    bst = xgb.train(params=params, dtrain=d_train, num_boost_round=num_boost, verbose_eval=False)
    run_time = round(time.time() - start_time, 2)
    bst.save_model(os.path.join(SAVE_PATH, f'0006_xgb_model_{num_boost}.model'))
    my_logger.info(f"num_boost_round: {num_boost} | time: {run_time}")
    return bst


def train_transfer_model(num_boost, model):
    d_train = xgb.DMatrix(train_x, label=train_y)

    params = {
        'booster': 'gbtree',
        'max_depth': 8,
        'min_child_weight': 7,
        'eta': 0.15,
        'objective': 'binary:logistic',
        'nthread': 5,
        'verbosity': 0,
        'eval_metric': 'logloss',
        'subsample': 0.5,
        'colsample_bytree': 0.5,
        'tree_method': 'hist',
        'seed': 1001,
    }

    start_time = time.time()
    bst = xgb.train(params=params,
                    dtrain=d_train,
                    num_boost_round=num_boost,
                    verbose_eval=False,
                    xgb_model=model)
    # print(evals_result)
    run_time = round(time.time() - start_time, 2)
    bst.save_model(os.path.join(SAVE_PATH, f'0006_xgb_model_transfer_{num_boost}.model'))
    my_logger.info(f"train_transfer_model: {num_boost} | base_model: {transfer_num_boost} | time: {run_time}")


if __name__ == '__main__':
    # train a global model
    # 100
    xgb_model = train_global_model(transfer_num_boost)
    # 300
    train_global_model(global_num_boost)
    # 100 + 200
    train_transfer_model(global_num_boost - transfer_num_boost, xgb_model)

    # load model
    # xgb_model_file = os.path.join(SAVE_PATH, f'0006_xgb_model_{transfer_num_boost}.model')
    # xgb_load_model = xgb.Booster(model_file=xgb_model_file)

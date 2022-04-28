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
import warnings
import time
import pickle
from my_logger import MyLog

# 自定义日志
my_logger = MyLog().logger

warnings.filterwarnings('ignore')

num_boost = 50
key_component = '24h_all_999_normalize'
train_x = pd.read_feather(f'/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/data/24h/{key_component}_train_x_data.feather')
train_y = pd.read_feather(f'/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/data/24h/{key_component}_train_y_data.feather')['Label']

test_x = pd.read_feather(f'/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/data/24h/{key_component}_test_x_data.feather')
test_y = pd.read_feather(f'/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/data/24h/{key_component}_test_y_data.feather')['Label']

# 迁移模型
xgb_model_file = '/panfs/pfs.local/work/liu/xzhang_sta/tangxizhuo/pg/global_model/0006_24h_xgb_glo4_div1_snap1_rm1_miss2_norm1.pkl'


def xgb_train_global(thread_num):
    d_train = xgb.DMatrix(train_x, label=train_y)
    d_test = xgb.DMatrix(test_x, label=test_y)

    params = {
        'booster': 'gbtree',
        'max_depth': 8,
        'min_child_weight': 7,
        'eta': 0.15,
        'objective': 'binary:logistic',
        'nthread': thread_num,
        'verbosity': 0,
        'eval_metric': 'logloss',
        'subsample': 0.5,
        'colsample_bytree': 0.5,
        'tree_method': 'hist',
        'seed': 1001,
    }

    evals_result = {}
    start_time = time.time()
    xgb.train(params=params,
              dtrain=d_train,
              evals=[(d_test, 'test')],
              num_boost_round=num_boost,
              evals_result=evals_result,
              verbose_eval=False,
              xgb_model=xgb_model)
    # print(evals_result)
    run_time = round(time.time() - start_time, 2)
    my_logger.info(f'thread_num: {thread_num} | train time: {run_time}')


if __name__ == '__main__':
    xgb_model = pickle.load(open(xgb_model_file, "rb"))
    for i in range(25):
        xgb_train_global(i)


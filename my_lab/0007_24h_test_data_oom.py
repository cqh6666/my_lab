# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     0007_24h_test_data_oom
   Description:   ...
   Author:        cqh
   date:          2022/4/25 14:03
-------------------------------------------------
   Change Activity:
                  2022/4/25:
-------------------------------------------------
"""
__author__ = 'cqh'
import pandas as pd
import os


# 训练集的X和Y
train_data_x_file = '/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/data/24h/24h_all_999_normalize_train_x_data.feather'
train_data_y_file = '/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/data/24h/24h_all_999_normalize_train_y_data.feather'
SAVE_PATH = '/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/result/personal_model_with_xgb/'
file_name = '0006_xgb_global_feature_weight_importance_boost91_v0.csv'

# ----- get data and init weight ----
train_x = pd.read_feather(train_data_x_file)
train_y = pd.read_feather(train_data_y_file)['Label']
normalize_weight = pd.read_csv(os.path.join(SAVE_PATH, file_name), index_col=None)
normalize_weight_sq = normalize_weight.squeeze()

# get data and init weight
key_component_name = 'div1_snap1_rm1_miss2_norm1'
train_x_2 = pd.read_feather(
    f'/panfs/pfs.local/work/liu/xzhang_sta/tangxizhuo/data/24/24h_train_x_{key_component_name}.feather')
train_y_2 = pd.read_feather(
    f'/panfs/pfs.local/work/liu/xzhang_sta/tangxizhuo/data/24/24h_train_y_{key_component_name}.feather')['Label']
normalize_weight_2 = pd.read_csv(
    f'/panfs/pfs.local/work/liu/xzhang_sta/tangxizhuo/pg/0008_24h_xgb_weight_glo2_{key_component_name}.csv',
    index_col=0).squeeze('columns')

print("=================================")
print("train_x")
print(train_x.shape)
print(train_x.head())
print("=================================")
print("train_x_2")
print(train_x_2.shape)
print(train_x_2.head())
print("=================================")
print("train_y")
print(train_y.shape)
print(train_y.head())
print("=================================")
print("train_y_2")
print(train_y_2.shape)
print(train_y_2.head())
print("=================================")
print("normalize_weight")
print(normalize_weight.shape)
print(normalize_weight.head())
print("=================================")
print("normalize_weight_sq")
print(normalize_weight_sq.shape)
print(normalize_weight_sq.head())
print("=================================")
print("normalize_weight_2")
print(normalize_weight_2.shape)
print(normalize_weight_2.head())
print("=================================")
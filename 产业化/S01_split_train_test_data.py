# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     S12_split_data
   Description:   ...
   Author:        cqh
   date:          2022/7/27 19:10
-------------------------------------------------
   Change Activity:
                  2022/7/27:
-------------------------------------------------
"""
__author__ = 'cqh'

from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# all_data = pd.read_csv("data_csv/default of credit card clients_new.csv", encoding='gbk')
all_data = pd.read_csv("data_csv/default of credit card clients.csv", encoding='gbk')
random_state = 2022

all_data_x = all_data.drop(['default payment next month'], axis=1)
all_data_y = all_data['default payment next month']

train_x, test_x, train_y, test_y = train_test_split(all_data_x, all_data_y, test_size=0.3, random_state=random_state)

train_data = pd.concat([train_x, train_y], axis=1)
test_data = pd.concat([test_x, test_y], axis=1)

train_data.to_csv("data_csv/raw_data/all_train_data.csv", index=False, encoding='utf-8')
test_data.to_csv("data_csv/raw_data/all_test_data.csv", index=False, encoding='utf-8')

# 标准化
norm_array = (all_data_x.abs().max().sort_values(ascending=False) > 100).index
min_max = MinMaxScaler()
all_data_x[norm_array] = pd.DataFrame(min_max.fit_transform(all_data_x[norm_array]), columns=norm_array)

all_data_x = all_data_x.drop(['ID'], axis=1)
train_x, test_x, train_y, test_y = train_test_split(all_data_x, all_data_y, test_size=0.3, random_state=random_state)

train_data = pd.concat([train_x, train_y], axis=1)
test_data = pd.concat([test_x, test_y], axis=1)

train_data.to_csv("data_csv/raw_data/all_train_data_norm.csv", index=False, encoding='utf-8')
test_data.to_csv("data_csv/raw_data/all_test_data_norm.csv", index=False, encoding='utf-8')


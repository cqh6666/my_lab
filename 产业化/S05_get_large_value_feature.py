# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     S06_get_max_value_feature
   Description:   ...
   Author:        cqh
   date:          2022/7/25 20:12
-------------------------------------------------
   Change Activity:
                  2022/7/25:
-------------------------------------------------
"""
__author__ = 'cqh'

from sklearn.model_selection import train_test_split
import pandas as pd

all_data = pd.read_csv("data_csv/default of credit card clients.csv")
all_data_x = all_data.drop(['default payment next month', 'ID'], axis=1)
all_data_y = all_data['default payment next month']

# train_x, test_x, train_y, test_y = train_test_split(all_data_x, all_data_y, test_size=0.3, random_state=2022)
norm_array = (all_data_x.abs().max().sort_values(ascending=False) > 100).index


# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     get_data
   Description:   ...
   Author:        cqh
   date:          2022/7/31 16:27
-------------------------------------------------
   Change Activity:
                  2022/7/31:
-------------------------------------------------
"""
__author__ = 'cqh'
import pandas as pd

data = pd.read_csv("./data_csv/default of credit card clients_new(Chinese).csv", encoding='gbk')

data_y = data['default payment next month']

data_y.value_counts(normalize=True)
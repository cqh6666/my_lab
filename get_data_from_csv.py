# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     get_data_from_csv
   Description:   ...
   Author:        cqh
   date:          2022/4/14 10:29
-------------------------------------------------
   Change Activity:
                  2022/4/14:
-------------------------------------------------
"""
__author__ = 'cqh'

import pandas as pd

csv_path = r"/panfs/pfs.local/work/liu/xzhang_sta/yaorunze/data/data/AKI_1_2_3/2016/24h/no_rolling/train_data.csv"
print("===========================begin================================")

data = pd.read_csv(csv_path)

print("shape:", data.shape)
print("columns:", data.columns)
print("head:", data.head())

print("===========================end================================")

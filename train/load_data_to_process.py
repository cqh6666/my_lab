# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     load_data_to_process
   Description:   ...
   Author:        cqh
   date:          2022/4/18 15:14
-------------------------------------------------
   Change Activity:
                  2022/4/18:
-------------------------------------------------
"""
__author__ = 'cqh'

import pandas as pd
import numpy as np
source_path = r"D:\lab\feather\iris_data.feather"

data = pd.read_feather(source_path)
data_X = data.iloc[:, :-1]
feature_happened = data_X > 0
feature_happened_count = feature_happened.sum()
feature_sum = data_X.sum()
feature_average_if = feature_sum / feature_happened_count
data_X = data_X / feature_average_if

print(range(1000))


a = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]])
b = np.array([[8,8,8]])
print(a-b)
print(np.mean(a-b))
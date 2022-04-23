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

csv_path = r"0006_xgb_global_feature_weight_importance_boost91_v0.csv"
print("===========================begin================================")

data = pd.read_csv(csv_path)
new_ki = []
for idx, value in enumerate(data.squeeze('columns')):
    print(idx, value)
    new_ki.append(value)

table = pd.DataFrame({'Ma_update_{}'.format(1): list(map(lambda x: x if x > 0 else 0, new_ki))})

print("===========================end================================")

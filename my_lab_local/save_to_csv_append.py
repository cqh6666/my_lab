# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     save_to_csv_append
   Description:   ...
   Author:        cqh
   date:          2022/5/19 19:39
-------------------------------------------------
   Change Activity:
                  2022/5/19:
-------------------------------------------------
"""
__author__ = 'cqh'

import pandas as pd

all_result = pd.DataFrame([[5, '0.5']], columns=['idx', 'auc'])
# all_result2 = pd.DataFrame([10, '0.5'], columns=['idx', 'auc'])
all_result.to_csv("24h_auc_result.csv", mode='a+', header=False, index=False)


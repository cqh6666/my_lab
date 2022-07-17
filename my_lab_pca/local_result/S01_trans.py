# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     S01_trans
   Description:   ...
   Author:        cqh
   date:          2022/7/16 16:12
-------------------------------------------------
   Change Activity:
                  2022/7/16:
-------------------------------------------------
"""
__author__ = 'cqh'

import pandas as pd

is_transfer = 1
csv_file_path = f"./csv/S01_xgb_auc_tra{is_transfer}.csv"
new_csv_file_path = f"./csv/S01_xgb_auc_tra{is_transfer}_v2.csv"
df = pd.read_csv(csv_file_path)
df.T.to_csv(new_csv_file_path, index=False)

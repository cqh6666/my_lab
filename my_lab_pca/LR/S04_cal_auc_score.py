# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     S02_cal_auc_score
   Description:   ...
   Author:        cqh
   date:          2022/7/17 13:29
-------------------------------------------------
   Change Activity:
                  2022/7/17:
-------------------------------------------------
"""
__author__ = 'cqh'
from sklearn.metrics import roc_auc_score
import pandas as pd

# S04_lr_test_tra1_mean20_v1.csv
version = 1
step = 4
test_result_file_name = f"./result/S04_mean_lr_test_v{version}.csv"

means = [20, 50, 100, 200]

result_df = pd.DataFrame(index=means, columns=['transfer', 'no_transfer'])
for comp in means:
    try:
        res = pd.read_csv(f"./result/S04_lr_test_tra1_mean{comp}_v1.csv")
        score = roc_auc_score(res['real'], res['prob'])
        result_df.loc[comp, 'transfer'] = score

        no_res = pd.read_csv(f"./result/S04_lr_test_tra0_mean{comp}_v1.csv")
        score = roc_auc_score(no_res['real'], no_res['prob'])
        result_df.loc[comp, 'no_transfer'] = score
    except Exception as err:
        continue

result_df.to_csv(test_result_file_name)

print(f"===================== step: S0{step}, version: {version} ======================================")
print(result_df)
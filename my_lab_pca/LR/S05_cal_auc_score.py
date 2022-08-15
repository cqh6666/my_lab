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

step = 5
version = 1
test_result_file_name = f"./result/S05_auto_encoder_lr_test_v{version}.csv"

pca_comps = "100"

result_df = pd.DataFrame(columns=['transfer', 'no_transfer'])
# S05_auto_encoder_lr_test_tra{is_transfer}_v{version}.csv
res = pd.read_csv(f"./result/S05_auto_encoder_lr_test_tra1_v{version}.csv")
no_res = pd.read_csv(f"./result/S05_auto_encoder_lr_test_tra0_v{version}.csv")

score = roc_auc_score(res['real'], res['prob'])
score2 = roc_auc_score(no_res['real'], no_res['prob'])
result_df.loc[pca_comps, 'transfer'] = score
result_df.loc[pca_comps, 'no_transfer'] = score2


result_df.to_csv(test_result_file_name)

print(f"===================== step: S0{step}, version: {version} ======================================")
print(result_df)
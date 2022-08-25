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

import os

from sklearn.metrics import roc_auc_score
import pandas as pd

step = 7
version = 1
# ./result/S06_temp/test_transfer/S07_test_iter0_dim100_tra1_v1.csv
tra1_dir_path = f"./result/S06_temp/test_transfer"
tra0_dir_path = f"./result/S06_temp/test_no_transfer"
test_result_file_name = f"./result/S07_test_dim100_v{version}.csv"

iter_list = [i for i in range(0, 130, 10)]

result_df = pd.DataFrame(columns=['transfer', 'no_transfer'])
for iter_idx in iter_list:
    try:
        res = pd.read_csv(os.path.join(tra1_dir_path, f"S0{step}_test_iter{iter_idx}_dim100_tra1_v{version}.csv"))
        no_res = pd.read_csv(os.path.join(tra0_dir_path, f"S0{step}_test_iter{iter_idx}_dim100_tra0_v{version}.csv"))

        score = roc_auc_score(res['real'], res['prob'])
        no_score = roc_auc_score(no_res['real'], no_res['prob'])
        result_df.loc[iter_idx, 'transfer'] = score
        result_df.loc[iter_idx, 'no_transfer'] = no_score
    except Exception as error:
        print(f"[{iter_idx}]find something error !")
        continue

result_df.to_csv(test_result_file_name)

print(f"===================== step: S0{step}, version: {version} ======================================")
print(result_df)

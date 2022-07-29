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

step = 3
version = 2
test_result_file_name = f"./result/S03_pca_xgb_test_v{version}.csv"

pca_comps = [20, 60, 100, 900, 920, 940, 960, 980, 1000]
spe_comps = [100, 900]

result_df = pd.DataFrame(index=pca_comps, columns=['transfer', 'no_transfer'])
for comp in pca_comps:
    res = pd.read_csv(f"./result/S03_pca_xgb_test_tra1_comp{comp}_v{version}.csv")
    no_res = pd.read_csv(f"./result/S03_pca_xgb_test_tra0_comp{comp}_v{version}.csv")

    if comp in spe_comps:
        res = res.iloc[:10001]
        no_res = no_res.iloc[:10001]

    score = roc_auc_score(res['real'], res['prob'])
    score2 = roc_auc_score(no_res['real'], no_res['prob'])
    result_df.loc[comp, 'transfer'] = score
    result_df.loc[comp, 'no_transfer'] = score2


result_df.to_csv(test_result_file_name)

print(f"===================== step: S0{step}, version: {version} ======================================")
print(result_df)
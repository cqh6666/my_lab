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

step = 2
version = 5
test_result_file_name = f"./result/S02_xgb_test_auc_result_v{version}.csv"

local_boost = [50]
select_rate = [10]

columns=['local_boost', 'seelct_rate', 'transfer', 'no_transfer']
result_df = pd.DataFrame(columns=columns)

for boost in local_boost:
    for select in select_rate:
        try:
            res = pd.read_csv(f"./result/S02_xgb_test_tra1_boost{boost}_select{select}_v{version}.csv")
            res2 = pd.read_csv(f"./result/S02_xgb_test_tra0_boost{boost}_select{select}_v{version}.csv")
            score = roc_auc_score(res['real'], res['prob'])
            score2 = roc_auc_score(res2['real'], res2['prob'])

            cur_res_df = pd.DataFrame([[boost, select, score, score2]], columns=columns)

            result_df = pd.concat([result_df, cur_res_df], ignore_index=True)

        except Exception as err:
            continue

result_df.to_csv(test_result_file_name, index=False)
print(f"===================== step:{step}, version:{version} ======================================")
print(result_df)
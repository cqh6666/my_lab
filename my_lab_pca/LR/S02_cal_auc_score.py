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
version = 11
test_result_file_name = f"./result/S02_lr_test_auc_result_v{version}.csv"

local_boost = [100]
select_rate = [10]

columns = ['local_boost', 'select_rate', 'transfer', 'no_transfer']
result_df = pd.DataFrame(columns=columns)

for boost in local_boost:
    for select in select_rate:
        try:
            # S02_lr_test_tra1_iter100_select10_v8.csv
            res = pd.read_csv(f"./result/S02_lr_test_tra1_iter{boost}_select{select}_v{version}.csv")
            res2 = pd.read_csv(f"./result/S02_lr_test_tra0_iter{boost}_select{select}_v{version}.csv")
            score = roc_auc_score(res['real'], res['prob'])
            score2 = roc_auc_score(res2['real'], res2['prob'])

            cur_res_df = pd.DataFrame([[boost, select, score, score2]], columns=columns)

            result_df = pd.concat([result_df, cur_res_df], ignore_index=True)

        except Exception as err:
            print(f"[{boost}, {select}] error!", err)
            continue

result_df.to_csv(test_result_file_name, index=False)
print(f"===================== step:{step}, version:{version} ======================================")
print(result_df)

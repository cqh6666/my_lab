# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     0009_cosine_psm
   Description:   ...
   Author:        cqh
   date:          2022/5/19 16:06
-------------------------------------------------
   Change Activity:
                  2022/5/19:
-------------------------------------------------
"""
__author__ = 'cqh'
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
import os
import pandas as pd


def get_psm_list():
    all_files = os.listdir(PSM_SAVE_PATH)
    cur_idx = 1
    all_weight = []
    for cur_file in all_files:
        if cur_idx >= max_idx:
            break
        file_flag = f"0008_24h_{cur_idx}_"
        if file_flag in cur_file:
            cur_weight = pd.read_csv(os.path.join(PSM_SAVE_PATH, f"{file_flag}feature_weight_localboost70_{transfer_flag}.csv")).squeeze()
            all_weight.append(cur_weight)
        cur_idx += step_idx

    return all_weight


if __name__ == '__main__':

    max_idx = 10
    step_idx = 1
    is_transfer = 0
    transfer_flag = "transfer" if is_transfer == 1 else "no_transfer"
    PSM_SAVE_PATH = f'/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/result/personal_model_with_xgb/24h_xgb_model/24h_{transfer_flag}_psm/'
    print(f"{transfer_flag} : cosine result: \n")
    all_weight_list = get_psm_list()
    result = cosine_similarity(all_weight_list)
    print(result)


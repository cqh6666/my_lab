# -*- coding: gbk -*-
"""
-------------------------------------------------
   File Name:     lr_utils_api
   Description:   ��ȡLR����ļ���Ϣ
   Author:        cqh
   date:          2022/7/8 14:41
-------------------------------------------------
   Change Activity:
                  2022/7/8:
-------------------------------------------------
"""
__author__ = 'cqh'

import pandas as pd
import os

from sklearn.linear_model import LogisticRegression

pre_hour = 24
root_dir = f"{pre_hour}h_old2"
DATA_SOURCE_PATH = f"/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/data/{root_dir}/"
MODEL_SAVE_PATH = f'/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/result/personal_model_with_lr/{root_dir}/global_model/'
global_lr_iter = 400


def get_init_similar_weight():
    init_similar_weight_file = os.path.join(MODEL_SAVE_PATH, f"0007_{pre_hour}h_0_psm_global_lr_{global_lr_iter}.csv")
    init_similar_weight = pd.read_csv(init_similar_weight_file).squeeze().tolist()
    return init_similar_weight


def get_transfer_weight(is_transfer):
    # ȫ��Ǩ�Ʋ��� ��Ҫ�õ���ʼ��csv
    if is_transfer == 1:
        init_weight_file_name = os.path.join(MODEL_SAVE_PATH, f"0007_{pre_hour}h_global_weight_lr_{global_lr_iter}.csv")
        global_feature_weight = pd.read_csv(init_weight_file_name).squeeze().tolist()
        return global_feature_weight
    else:
        return None

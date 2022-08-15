# -*- coding: gbk -*-
"""
-------------------------------------------------
   File Name:     lr_utils_api
   Description:   获取LR相关文件信息
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


pre_hour = 24
MODEL_SAVE_PATH = f"/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/code_pca/code_lr/result/global_model/"
global_lr_iter = 400


def get_init_similar_weight():
    init_similar_weight_file = os.path.join(MODEL_SAVE_PATH, f"0007_{pre_hour}h_0_psm_global_lr_{global_lr_iter}.csv")
    init_similar_weight = pd.read_csv(init_similar_weight_file).squeeze().tolist()
    return init_similar_weight


def get_lr_init_similar_weight():
    return get_init_similar_weight()

def get_transfer_weight(is_transfer):
    # 全局迁移策略 需要用到初始的csv
    if is_transfer == 1:
        init_weight_file_name = os.path.join(MODEL_SAVE_PATH, f"0007_{pre_hour}h_global_weight_lr_{global_lr_iter}.csv")
        global_feature_weight = pd.read_csv(init_weight_file_name).squeeze().tolist()
        return global_feature_weight
    else:
        return None


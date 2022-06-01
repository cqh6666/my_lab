# -*- coding: gbk -*-
"""
-------------------------------------------------
   File Name:     0008_add_psm_feature_name
   Description:   ...
   Author:        cqh
   date:          2022/5/23 16:20
-------------------------------------------------
   Change Activity:
                  2022/5/23:
-------------------------------------------------
"""
__author__ = 'cqh'

import os
import sys
import pandas as pd


def get_init_psm_file():
    # transfer and no transfer
    init_psm_file_name = '0006_xgb_global_feature_weight_boost100.csv'
    init_psm_file = os.path.join(INIT_PSM_PATH, init_psm_file_name)
    return init_psm_file


def get_iter_psm_flie():
    # 某次迭代
    if is_transfer == 0:
        iter_psm_file_name = f'0008_24h_{iter_idx}_feature_weight_localboost70_no_transfer.csv'
    else:
        iter_psm_file_name = f'0008_24h_{iter_idx}_feature_weight_gtlboost20_localboost50.csv'
    iter_psm_file = os.path.join(ITER_PSM_PATH, iter_psm_file_name)
    return iter_psm_file


def run():

    if iter_idx == 0:
        psm_file = get_init_psm_file()
    else:
        psm_file = get_iter_psm_flie()

    psm_csv_file = pd.read_csv(psm_file)
    # 特征字典
    remained_feature_explain = pd.read_csv(os.path.join(FEATURE_MAP_PATH, "remained_feature_explain.csv"))

    concat_psm_explain = pd.concat([psm_csv_file, remained_feature_explain], axis=1, ignore_index=True)
    concat_psm_explain.columns = ['psm_value', 'new_feature', 'VAR_IDX', 'VAR_POS', 'TABLE_NAME', 'FIELD_NAME', 'VALUESET_ITEM', 'VALUESET_ITEM_DESCRIPTOR']
    concat_psm_explain.to_csv(os.path.join(SAVE_PSM_PATH, f"0008_24h_{iter_idx}_psm_explain_{transfer_flag}.csv"), index=False)
    print("done!")


if __name__ == '__main__':

    iter_idx = int(sys.argv[1])
    is_transfer = int(sys.argv[2])

    transfer_flag = "transfer" if is_transfer == 1 else "no_transfer"

    INIT_PSM_PATH = '/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/result/personal_model_with_xgb/24h_xgb_model/'
    ITER_PSM_PATH = f'/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/result/personal_model_with_xgb/24h_xgb_model/24h_{transfer_flag}_psm/'
    FEATURE_MAP_PATH = "/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/data/"
    SAVE_PSM_PATH = '/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/result/personal_model_with_xgb/24h_xgb_model/temp_output/'

    run()


















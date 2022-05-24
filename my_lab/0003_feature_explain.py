# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     feature_explain
   Description:   ...
   Author:        cqh
   date:          2022/5/23 14:13
-------------------------------------------------
   Change Activity:
                  2022/5/23:
-------------------------------------------------
"""
__author__ = 'cqh'

import os

from my_logger import MyLog
import pandas as pd


def run():
    """
    根据特征字典和新旧特征 对筛选后的特征进行对应的解释 主要是 lab,ccs,px,med
    range:
        LAB_1 ~ LAB_817
        CCS_1 ~ CCS_99
        PX_1 ~ PX_15606
        MED_1 ~ MED_9999
    :return:
    """
    old_and_new_feature_map = pd.read_csv(os.path.join(FEATURE_MAP_PATH, "old_and_new_feature_map.csv"))
    feature_dict = pd.read_csv(os.path.join(FEATURE_MAP_PATH, "feature_dict.csv"))
    # 筛选后的特征
    remained_new_feature_map = pd.read_csv(os.path.join(FEATURE_MAP_PATH, "24_999_remained_new_feature_map.csv"),
                                           header=None)

    remained_new_feature_list = remained_new_feature_map.squeeze().tolist()
    # 去除 ID Label
    remained_new_feature_list.remove('ID')
    remained_new_feature_list.remove('Label')

    explain_columns = feature_dict.columns
    remained_feature_explain = pd.DataFrame(columns=explain_columns)
    select_prefix = ['LA', 'CC', 'PX', 'ME']

    for cur_feature in remained_new_feature_list:
        prefix = cur_feature[:2]
        if prefix in select_prefix:
            old_feature = old_and_new_feature_map.loc[old_and_new_feature_map['new_feature'] == cur_feature]['old_feature'].iloc[0]
            feature_select_list = feature_dict[feature_dict['VAR_IDX'] == old_feature]
            # 没找到特征映射
            if len(feature_select_list) == 0:
                remained_feature_explain.loc[cur_feature] = ""
            else:
                remained_feature_explain.loc[cur_feature] = feature_select_list.iloc[0]
        else:
            # 不需要解释
            remained_feature_explain.loc[cur_feature] = ""
            continue
    remained_feature_explain.to_csv(os.path.join(FEATURE_MAP_PATH, "remained_feature_explain.csv"))


if __name__ == '__main__':
    FEATURE_MAP_PATH = "/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/data/"
    my_logger = MyLog().logger
    run()

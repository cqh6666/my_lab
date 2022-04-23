# -*- coding: gbk -*-
"""
-------------------------------------------------
   File Name:     0004_24h_analysis_data.py
   Description:   ...
   Author:        cqh
   date:          2022/4/20 14:31
-------------------------------------------------
   Change Activity:
                  2022/4/20:
-------------------------------------------------
"""
__author__ = 'cqh'
import pandas as pd

SOURCE_FILE_PATH = '/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/data/24h/all_24h_dataframe_999_feature_normalize.feather'


def get_data():
    all_samples = pd.read_feather(SOURCE_FILE_PATH)

    all_samples_y = all_samples['Label']

    print("==================================")
    print("各类分布:")
    print(all_samples_y.value_counts())
    print("==================================")



if __name__ == '__main__':
    get_data()
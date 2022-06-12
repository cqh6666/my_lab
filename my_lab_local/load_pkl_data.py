# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     load_pkl_data
   Description:   ...
   Author:        cqh
   date:          2022/5/30 12:19
-------------------------------------------------
   Change Activity:
                  2022/5/30:
-------------------------------------------------
"""
__author__ = 'cqh'

import joblib

# string_list_file_path = "D:\\lab\\other_file\\2016_string2list.pkl"
# list_data = joblib.load(string_list_file_path)
# print(".")

import pandas as pd

csv = pd.read_csv(r"D:\lab\other_file\0008_24h_21_psm_transfer.csv")
print(".")
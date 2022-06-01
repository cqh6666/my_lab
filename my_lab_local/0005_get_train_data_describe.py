# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     0005_get_all_data_mean_std
   Description:   ...
   Author:        cqh
   date:          2022/5/23 9:33
-------------------------------------------------
   Change Activity:
                  2022/5/23:
-------------------------------------------------
"""
__author__ = 'cqh'
import pandas as pd
import os


def get_data_describe(train_data):
    data_describe = train_data.describe().T
    data_describe.index = train_data.columns
    file_name = os.path.join(SAVE_PATH, result_file)
    data_describe.to_csv(file_name)
    print("save csv success!", file_name)


if __name__ == '__main__':

    pre_hour = 24
    DATA_SOURCE_PATH = r'D:\dataset\mushroom'
    SAVE_PATH = r'D:\dataset\mushroom'
    file_flag = f"{pre_hour}_df_rm1_norm1"
    result_file = f"train_data_describe_info_{file_flag}.csv"
    train_x = pd.read_csv(os.path.join(DATA_SOURCE_PATH, r"mushroom_out.csv"))
    print(train_x.info())
    get_data_describe(train_x)


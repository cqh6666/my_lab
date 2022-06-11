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

    train_row = train_data.shape[0]
    zero_counts = train_data.apply(lambda x: x.value_counts().get(0, 0), axis=0).tolist()
    zero_counts_percent = [zero / train_row for zero in zero_counts]
    zero_df = pd.DataFrame({"zero_percent": zero_counts_percent}, index=train_data.columns)
    data_describe = pd.concat([data_describe, zero_df], axis=1)

    data_describe.to_csv(save_file_name)
    print("save csv success!", save_file_name)
    print(data_describe.head())


if __name__ == '__main__':

    pre_hour = 24
    root_dir = f"{pre_hour}h_old2"
    DATA_SOURCE_PATH = f'/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/data/{root_dir}/'

    # 训练集数据
    file_flag = f"{pre_hour}_df_rm1_norm1"
    train_x = pd.read_feather(os.path.join(DATA_SOURCE_PATH, f"all_x_train_{file_flag}.feather"))

    result_file = f"train_x_describe_info_{file_flag}.csv"
    save_file_name = os.path.join(DATA_SOURCE_PATH, result_file)

    get_data_describe(train_x)
    print(train_x.info)



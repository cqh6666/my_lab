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


def get_data_describe(train_data, result_file):
    data_describe = train_data.describe().T
    data_describe.index = train_data.columns

    train_row = train_data.shape[0]
    zero_counts = train_data.apply(lambda x: x.value_counts().get(0, 0), axis=0).tolist()
    zero_counts_percent = [zero / train_row for zero in zero_counts]
    zero_df = pd.DataFrame({"zero_percent": zero_counts_percent}, index=train_data.columns)
    data_describe = pd.concat([data_describe, zero_df], axis=1)

    file_name = os.path.join(SAVE_PATH, result_file)
    data_describe.to_csv(file_name)
    print("save csv success!", file_name)
    print(data_describe.head())


if __name__ == '__main__':

    pre_hour = 24
    DATA_SOURCE_PATH = f'/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/data/{pre_hour}h/'
    SAVE_PATH = f'/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/data/code/'
    file_flag = f"{pre_hour}_df_rm1_norm1"
    result_file = f"all_data_describe_info_{file_flag}.csv"
    # all_24h_norm_dataframe_999_miss_medpx_max2dist.feather
    train_x = pd.read_feather(os.path.join(DATA_SOURCE_PATH, f"all_24h_norm_dataframe_999_miss_medpx_max2dist.feather"))
    get_data_describe(train_x, result_file)
    print(train_x.info)

    t_data_file = "/panfs/pfs.local/work/liu/xzhang_sta/tangxizhuo/data/24/all_24h_snap1_rm2_miss2_norm1.feather"
    t_result_file = f"all_data_describe_info_snap1_rm2_miss2_norm1.csv"
    t_data = pd.read_feather(t_data_file)
    get_data_describe(t_data, t_result_file)
    print(t_data.info)


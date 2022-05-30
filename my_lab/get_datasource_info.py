# -*- coding: gbk -*-
"""
-------------------------------------------------
   File Name:     get_datasource_info
   Description:   ...
   Author:        cqh
   date:          2022/5/27 9:17
-------------------------------------------------
   Change Activity:
                  2022/5/27:
-------------------------------------------------
"""
__author__ = 'cqh'

import pandas as pd
import os


def get_train_info(file_name, file_type='csv', file_flag='yuan'):
    """ main, std, zero_counts"""
    if file_type == 'csv':
        train_df = pd.read_csv(file_name)
    elif file_type == 'feather':
        train_df = pd.read_feather(file_name)
    else:
        print("no found file type...")
        return

    train_row = train_df.shape[0]
    mean_list = train_df.mean().tolist()
    std_list = train_df.std().tolist()
    zero_counts = train_df.apply(lambda x: x.value_counts().get(0, 0), axis=0).tolist()

    zero_counts_percent = [zero / train_row for zero in zero_counts]
    columns = train_df.columns.tolist()

    result = pd.DataFrame({"mean": mean_list, "std": std_list, "zero_counts": zero_counts, "zero_percent": zero_counts_percent}, index=columns)
    save_name = os.path.join(SAVE_PATH, f"{file_flag}_data_mean_std_zero.csv")
    result.to_csv(save_name)
    print(f"save success! - {save_name}")


def get_train_test_data():
    DATA_SOURCE_PATH = f"/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/data/24h/"

    train_x = pd.read_feather(
        os.path.join(DATA_SOURCE_PATH, "all_x_train_24_df_rm1_miss2_norm2.feather"))
    train_y = pd.read_feather(
        os.path.join(DATA_SOURCE_PATH, "all_y_train_24_df_rm1_miss2_norm2.feather"))['Label']
    test_x = pd.read_feather(
        os.path.join(DATA_SOURCE_PATH, "all_x_test_24_df_rm1_miss2_norm2.feather"))
    test_y = pd.read_feather(
        os.path.join(DATA_SOURCE_PATH, "all_y_test_24_df_rm1_miss2_norm2.feather"))['Label']

    return train_x, train_y, test_x, test_y


if __name__ == '__main__':
    SAVE_PATH = "/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/learn/csv_output/"
    DATA_SOURCE_PATH = f"/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/data/24h/"
    yuan_file_name = '/panfs/pfs.local/work/liu/xzhang_sta/yuanborong/data/df_data/pre_48h/train-data-no-empty-normalized.csv'
    hai_file_name = os.path.join(DATA_SOURCE_PATH, "all_x_train_24_df_rm1_miss2_norm2.feather")
    hai_file_name2 = os.path.join(DATA_SOURCE_PATH, "all_x_train_24_df_rm1_norm2.feather")
    # 只对lab标准化，不对med,px标准化
    hai_file_name3 = os.path.join(DATA_SOURCE_PATH, "all_x_train_24h_norm_dataframe_999_miss_medpx_max2dist.feather")

    get_train_info(hai_file_name, 'feather', file_flag='hai_miss2_norm2')
    get_train_info(hai_file_name2, 'feather', file_flag='hai_norm2')
    get_train_info(hai_file_name3, 'feather', file_flag='hai_norm1')

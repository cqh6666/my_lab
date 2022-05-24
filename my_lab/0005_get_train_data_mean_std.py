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


def get_mean_std(train_data):
    train_mean = train_data.mean().tolist()
    train_std = train_data.std().tolist()

    feature_mean_std = pd.DataFrame({"std": train_std, "mean": train_mean}, index=train_data.columns)

    file_name = os.path.join(SAVE_PATH, "train_data_feature_mean_std.csv")
    feature_mean_std.to_csv(file_name)
    print("save csv success!", file_name)


if __name__ == '__main__':

    DATA_SOURCE_PATH = f"/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/data/24h/"  # 训练集的X和Y
    SAVE_PATH = f'/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/result/personal_model_with_xgb/24h_xgb_model/'
    train_x = pd.read_feather(
        os.path.join(DATA_SOURCE_PATH, "all_x_train_24h_norm_dataframe_999_miss_medpx_max2dist.feather"))

    get_mean_std(train_x)


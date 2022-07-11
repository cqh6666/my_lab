# -*- coding: gbk -*-
"""
-------------------------------------------------
   File Name:     utils_api
   Description:   ...
   Author:        cqh
   date:          2022/7/5 10:15
-------------------------------------------------
   Change Activity:
                  2022/7/5:
-------------------------------------------------
"""
__author__ = 'cqh'

import pandas as pd
import os
from sklearn.model_selection import train_test_split
import pickle

pre_hour = 24
root_dir = f"{pre_hour}h_old2"
DATA_SOURCE_PATH = f"/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/data/{root_dir}/"
MODEL_SAVE_PATH = f'/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/result/psm_with_xgb/{root_dir}/global_model/'

all_data_file = f"all_{pre_hour}_df_rm1_norm1.feather"


def get_train_test_data(test_size=0.15):
    """
    得到完整的数据 train_data, test_data
    :param test_size:
    :return:
    """
    all_data_path = os.path.join(DATA_SOURCE_PATH, all_data_file)
    all_samples = pd.read_feather(all_data_path)
    train_data, test_data = train_test_split(all_samples, test_size=test_size, random_state=2022)
    return train_data, test_data


def get_train_x_y_data():
    train_data, _ = get_train_test_data()
    train_data_x = train_data.drop(['ID', 'Label'], axis=1)
    train_data_y = train_data['Label']
    return train_data_x, train_data_y


def covert_time_format(seconds):
    """将秒数转成比较好显示的格式
    >>> covert_time_format(3600) == '1.0 h'
    True
    >>> covert_time_format(360) == '6.0 m'
    True
    >>> covert_time_format(6) == '36 s'
    True
    """
    assert isinstance(seconds, (int, float))
    hour = seconds // 3600
    if hour > 0:
        return f"{round(hour + seconds % 3600 / 3600, 2)} h"

    minute = seconds // 60
    if minute > 0:
        return f"{round(minute + seconds % 60 / 60, 2)} m"

    return f"{round(seconds, 2)} s"


if __name__ == '__main__':
    print(covert_time_format(12323))
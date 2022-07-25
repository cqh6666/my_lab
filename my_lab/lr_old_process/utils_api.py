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

pre_hour = 24
root_dir = f"{pre_hour}h_old2"
DATA_SOURCE_PATH = f"/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/data/{root_dir}/"

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
    train_data.reset_index(drop=True, inplace=True)
    test_data.reset_index(drop=True, inplace=True)

    return train_data, test_data


def get_train_test_x_y():
    """
    得到训练集和测试集，并重置ID，节省空间
    :return:
    """
    train_data, test_data = get_train_test_data()

    train_data_x = train_data.drop(['Label', 'ID'], axis=1)
    train_data_y = train_data['Label']

    test_data_x = test_data.drop(['Label', 'ID'], axis=1)
    test_data_y = test_data['Label']

    return train_data_x, train_data_y, test_data_x, test_data_y


def get_target_test_id():
    """
    得到50个正样本，50个负样本来进行分析
    :return:
    """
    _, test_data = get_train_test_data()
    test_data_y = test_data['Label']
    test_data_ids_1 = test_data_y[test_data_y == 1].index[:50].values
    test_data_ids_0 = test_data_y[test_data_y == 0].index[:50].values

    return test_data_ids_1, test_data_ids_0



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


def save_to_csv_by_row(csv_file, new_df):
    """
    以行的方式插入csv文件之中，若文件存在则在尾行插入，否则新建一个新的csv；
    :param csv_file: 默认保存的文件
    :param new_df: dataFrame格式 需要包含header
    :return:
    """
    # 保存存入的是dataFrame格式
    assert isinstance(new_df, pd.DataFrame)
    # 不能存在NaN
    if new_df.isna().sum().sum() > 0:
        print("exist NaN...")
        return False

    if os.path.exists(csv_file):
        new_df.to_csv(csv_file, mode='a', index=True, header=False)
    else:
        new_df.to_csv(csv_file, index=True, header=True)

    return True


if __name__ == '__main__':
    test_1, test_0 = get_target_test_id()
    print("test_1", test_1)
    print("test_0", test_0)




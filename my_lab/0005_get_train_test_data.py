# encoding=gbk
"""
input:
    {year}_24h_dataframe_999_feature_normalize.feather
output:
    24h_train_x_by_year.feather
    24h_train_y_by_year.feather
    24h_test_x_by_year.feather
    24h_test_y_by_year.feather
"""
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from my_logger import MyLog


def get_data_from_feather_to_save(test_size=0.15):
    load_data_file = os.path.join(SOURCE_FILE_PATH, f"all_{miss_norm_file_name}.feather")
    all_samples = pd.read_feather(load_data_file)
    my_logger.info(f"all samples: {all_samples.shape}")
    all_samples_y = all_samples['Label']
    all_samples_x = all_samples.drop(['ID', 'Label'], axis=1)

    x_train, x_test, y_train, y_test = train_test_split(all_samples_x, all_samples_y, test_size=test_size)

    my_logger.info(f"x_train shape: {x_train.shape} | y_train_shape: {y_train.shape}")
    my_logger.info(f"x_test_shape: {x_test.shape} | y_test_shape: {y_test.shape}")

    # save feather
    save_dataFrame_to_feather(x_train, y_train, "train")
    save_dataFrame_to_feather(x_test, y_test, "test")


def save_dataFrame_to_feather(x_data, y_data, file_flag):
    """
    ??ѵ????X,Y????Ϊfeather
    :param x_data:
    :param y_data:
    :param file_flag:
    :return:
    """
    x_data.reset_index(drop=True, inplace=True)
    y_data.reset_index(drop=True, inplace=True)
    load_x_file = os.path.join(SOURCE_FILE_PATH, f"all_x_{file_flag}_{miss_norm_file_name}.feather")
    load_y_file = os.path.join(SOURCE_FILE_PATH, f"all_y_{file_flag}_{miss_norm_file_name}.feather")
    x_data.to_feather(load_x_file)
    y_data.to_frame(name='Label').to_feather(load_y_file)

    my_logger.info(f"save x_{file_flag} file - [{load_x_file}]")
    my_logger.info(f"save y_{file_flag} file - [{load_y_file}]")


if __name__ == '__main__':
    pre_hour = 24

    SOURCE_FILE_PATH = f'/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/data/{pre_hour}h/'

    miss_norm_file_name = f"{pre_hour}_df_rm1_norm2"

    my_logger = MyLog().logger
    get_data_from_feather_to_save()


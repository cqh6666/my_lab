#encoding=gbk
"""
input:
    {year}_24h_snap{id}_rm{id}.feather
    all_24h_snap{id}_rm{id}.feather
output:
    0004_LAB_521_with_negative.log
    all_24h_snap{id}_rm{id}_miss{id}_norm{id}.feather
    {year}_24h_snap{id}_rm{id}_miss{id}_norm{id}.feather
"""
import pandas as pd
import numpy as np
from my_logger import MyLog
import os


def get_norm1_feature_list():
    """
    收集所有需要标准化的特征列表
    collect features needed to be normalized: age, height, weight, bmi, sbp, dbp, all lab
    :return:
    """
    normalize_feature_list = ['DEMO_Age', 'VITAL_Height', 'VITAL_Weight', 'VITAL_BMI', 'VITAL_SBP', 'VITAL_DBP']
    # load 24h remained feature from csv file
    for feature in remained_feature_list:
        if feature.startswith('LAB'):
            normalize_feature_list.append(feature)
    my_logger.info(f"normalize feature list len: {len(normalize_feature_list)}")
    return normalize_feature_list


def get_norm2_feature_list():
    """
    收集所有需要标准化的特征列表
    collect features needed to be normalized: age, height, weight, bmi, sbp, dbp, all lab, all px, all med
    :return:
    """
    normalize_feature_list = ['DEMO_Age', 'VITAL_Height', 'VITAL_Weight', 'VITAL_BMI', 'VITAL_SBP', 'VITAL_DBP']
    # load 24h remained feature from csv file
    for feature in remained_feature_list:
        if feature.startswith('LAB') or feature.startswith('MED') or feature.startswith('PX'):
            normalize_feature_list.append(feature)
    my_logger.info(f"normalize feature list len: {len(normalize_feature_list)}")
    return normalize_feature_list


def get_average_if(source_data, norm_feature_list):
    """calculate feature average if by all samples"""
    all_sample = source_data
    normalize_feature_list = norm_feature_list
    feature_happen_count = (all_sample.loc[:, normalize_feature_list] != 0).sum(axis=0)
    feature_sum = all_sample.loc[:, normalize_feature_list].sum(axis=0)
    feature_average_if = feature_sum / feature_happen_count
    return feature_average_if


def norm_feature_all(miss_file, miss_norm_file):
    """normalize specified features for all samples"""

    load_file_name = os.path.join(DATA_SOURCE_PATH, all_rm999_process_file_name)
    # load_file_name = os.path.join(DATA_SOURCE_PATH, f"all_{miss_file}.feather")
    save_file_name = os.path.join(DATA_SOURCE_PATH, f"all_{miss_norm_file}.feather")

    all_sample = pd.read_feather(load_file_name)

    # norm列表
    normalize_feature_list = get_norm2_feature_list()
    feature_average_if = get_average_if(all_sample, normalize_feature_list)
    all_sample.loc[:, normalize_feature_list] = all_sample.loc[:, normalize_feature_list] / feature_average_if

    # save
    all_sample.to_feather(save_file_name)
    my_logger.info(f"save feather data success! - [{save_file_name}]")


def get_med_px_feature_list():
    """
    获取procedure(px) med 所有特征列表
    :return:
    """
    med_px_feature_list = []
    # load 24h remained feature from csv file
    for feature in remained_feature_list:
        if feature.startswith('MED') or feature.startswith('PX'):
            med_px_feature_list.append(feature)

    return med_px_feature_list


def get_med_px_max_distance():
    """
    最大距离
    :return:
    """
    load_file_name = os.path.join(DATA_SOURCE_PATH, all_rm999_process_file_name)

    all_sample = pd.read_feather(load_file_name)
    med_px_feature_list = get_med_px_feature_list()
    med_px_all_sample = all_sample.loc[:, med_px_feature_list]
    my_logger.info(f"get med and px max distance success!")
    return med_px_all_sample.max(axis=0) * 2


def fill_miss2_all(file_name):
    """---miss 2---
    将med和px标签值为0设为该标签最大值*2
    med and px: 2 * max
    :param file_name: 保存文件名称
    :return:
    """
    # set read file and save file
    load_file_name = os.path.join(DATA_SOURCE_PATH, all_rm999_process_file_name)
    save_file_name = os.path.join(DATA_SOURCE_PATH, f"all_{file_name}.feather")

    # get the remained med and px feature name
    med_px_feature_list = get_med_px_feature_list()
    max_time_distance = get_med_px_max_distance()

    # load all sample
    all_sample = pd.read_feather(load_file_name)

    # focus on med and px
    med_px_all_sample = all_sample.loc[:, med_px_feature_list]

    # for med and px, fill 0 with 2 * max
    bool_mask = (med_px_all_sample == 0)
    all_sample.loc[:, med_px_feature_list] = med_px_all_sample.mask(bool_mask, max_time_distance, axis=1)

    # save file
    all_sample.to_feather(save_file_name)
    my_logger.info(f"save to feather after missing process.. - [{save_file_name}]")


def norm1_feature_each_year():
    """normalize specified features for each year"""

    normalize_feature_list = get_norm1_feature_list()
    feature_average_if = get_average_if(
        '/panfs/pfs.local/work/liu/xzhang_sta/tangxizhuo/data/24/all_24h_snap1_rm1_miss2.feather', normalize_feature_list)

    # traverse each year and normalize feature
    for year in range(2010, 2018 + 1):
        cur_data = pd.read_feather(
            f"/panfs/pfs.local/work/liu/xzhang_sta/tangxizhuo/data/{pre_hour}/{year}_{pre_hour}h_snap1_rm1_miss2.feather")
        cur_data.loc[:, normalize_feature_list] = cur_data.loc[:, normalize_feature_list] / feature_average_if
        # check
        cur_data.loc[:, 'DEMO_Age'].to_csv(f'{year}check.csv')
        cur_data.to_feather(
            f"/panfs/pfs.local/work/liu/xzhang_sta/tangxizhuo/data/{pre_hour}/{year}_{pre_hour}h_snap1_rm1_miss2_norm1.feather")


def fill_miss2_each_year(file_name, hour=24):
    med_px_feature_list = get_med_px_feature_list()
    max_time_distance = get_med_px_max_distance()

    # traverse each year and normalize feature
    for year in range(2010, 2018 + 1):
        cur_data = pd.read_feather(
            f"/panfs/pfs.local/work/liu/xzhang_sta/tangxizhuo/data/24/{year}_24h_snap1_rm1.feather")
        cur_med_px_data = cur_data.loc[:, med_px_feature_list]
        bool_mask = (cur_med_px_data == 0)
        cur_data.loc[:, med_px_feature_list] = cur_med_px_data.mask(bool_mask, max_time_distance, axis=1)
        # save file
        cur_data.to_feather(
            f"/panfs/pfs.local/work/liu/xzhang_sta/tangxizhuo/data/{pre_hour}/{year}_24h_snap1_rm1_miss2.feather")


def fill_miss3_each_year():
    med_px_feature_list = get_med_px_feature_list()

    # traverse each year and normalize feature
    for year in range(2010, 2018 + 1):
        cur_data = pd.read_feather(
            f"/panfs/pfs.local/work/liu/xzhang_sta/tangxizhuo/data/{pre_hour}/{year}_24h_snap1_rm1_miss1_norm1.feather")
        cur_med_px_data = cur_data.loc[:, med_px_feature_list]
        cur_med_px_data[cur_med_px_data == 0] = np.nan
        cur_data.loc[:, med_px_feature_list] = cur_med_px_data
        # save file
        cur_data.to_feather(
            f"/panfs/pfs.local/work/liu/xzhang_sta/tangxizhuo/data/{pre_hour}/{year}_24h_snap1_rm1_miss3_norm1.feather")


if __name__ == '__main__':

    pre_hour = 24

    FEATURE_MAP_PATH = "/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/data/"
    DATA_SOURCE_PATH = f"/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/data/{pre_hour}h/"

    # 初始处理数据
    all_rm999_process_file_name = f"all_{pre_hour}h_dataframe_999_feature_remove.feather"

    miss_process_file_name = f"{pre_hour}_df_rm1_miss2"
    miss_norm_process_file_name = f"{pre_hour}_df_rm1_norm2"

    remained_feature_file = os.path.join(FEATURE_MAP_PATH, f'{pre_hour}_999_remained_new_feature_map.csv')
    remained_feature_list = pd.read_csv(remained_feature_file, header=None).squeeze().tolist()

    my_logger = MyLog().logger

    # 对px和med特征miss值处理
    # fill_miss2_all(miss_process_file_name)
    norm_feature_all(miss_process_file_name, miss_norm_process_file_name)

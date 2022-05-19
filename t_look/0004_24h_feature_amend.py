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


def get_norm1_feature_list():
    """collect features needed to be normalized: age, height, weight, bmi, sbp, dbp, all lab"""

    normalize_feature_list = ['DEMO_Age', 'VITAL_Height', 'VITAL_Weight', 'VITAL_BMI', 'VITAL_SBP', 'VITAL_DBP']
    # load 24h remained feature from csv file
    remained_feature_name = pd.read_csv('/panfs/pfs.local/work/liu/xzhang_sta/tangxizhuo/data/24/24h_snap1_rm1_remain_feature.csv',
                                        header=None).squeeze("columns").tolist()
    for feature in remained_feature_name:
        if feature.startswith('LAB'):
            normalize_feature_list.append(feature)

    return normalize_feature_list


def check_negative_feature():

    all_sample = pd.read_feather('/panfs/pfs.local/work/liu/xzhang_sta/tangxizhuo/data/24/all_24h_snap1_rm1.feather')
    print((all_sample < 0).values.sum())
    row = all_sample.index[np.where(all_sample < 0)[0]].tolist()
    column = all_sample.columns[np.where(all_sample < 0)[1]].tolist()
    points = list(zip(row, column))
    for x, y in points:
        print(f"index-{x}, column-{y}: {all_sample.loc[x, y]}")


def check_lab521():

    all_sample = pd.read_feather('/panfs/pfs.local/work/liu/xzhang_sta/tangxizhuo/data/24/all_24h_snap1_rm1.feather')
    print((all_sample.loc[:, 'LAB_521'] > 0).values.sum())
    print("test value LAB_521: ", all_sample.loc[80386, 'LAB_521'])
    print("test value other: ", all_sample.loc[80386, 'ID'])


def get_average_if(load_file):
    """calculate feature average if by all samples"""

    all_sample = pd.read_feather(load_file)
    normalize_feature_list = get_norm1_feature_list()
    feature_happen_count = (all_sample.loc[:, normalize_feature_list] != 0).sum(axis=0)
    feature_sum = all_sample.loc[:, normalize_feature_list].sum(axis=0)
    feature_average_if = feature_sum / feature_happen_count
    return feature_average_if


def norm1_feature_all():
    """normalize specified features for all samples"""

    load_file = '/panfs/pfs.local/work/liu/xzhang_sta/tangxizhuo/data/24/all_24h_snap1_rm1_miss2.feather'
    save_file = '/panfs/pfs.local/work/liu/xzhang_sta/tangxizhuo/data/24/all_24h_snap1_rm1_miss2_norm1.feather'
    normalize_feature_list = get_norm1_feature_list()
    feature_average_if = get_average_if(load_file)

    all_sample = pd.read_feather(load_file)
    all_sample.loc[:, normalize_feature_list] = all_sample.loc[:, normalize_feature_list] / feature_average_if
    # save all samples to one file after normalizing feature
    all_sample.to_feather(save_file)


def norm1_feature_each_year():
    """normalize specified features for each year"""

    feature_average_if = get_average_if(
        '/panfs/pfs.local/work/liu/xzhang_sta/tangxizhuo/data/24/all_24h_snap1_rm1_miss2.feather')
    normalize_feature_list = get_norm1_feature_list()

    # traverse each year and normalize feature
    for year in range(2010, 2018 + 1):
        cur_data = pd.read_feather(
            f"/panfs/pfs.local/work/liu/xzhang_sta/tangxizhuo/data/24/{year}_24h_snap1_rm1_miss2.feather")
        cur_data.loc[:, normalize_feature_list] = cur_data.loc[:, normalize_feature_list] / feature_average_if
        # check
        cur_data.loc[:, 'DEMO_Age'].to_csv(f'{year}check.csv')
        cur_data.to_feather(
            f"/panfs/pfs.local/work/liu/xzhang_sta/tangxizhuo/data/24/{year}_24h_snap1_rm1_miss2_norm1.feather")


def get_med_px_feature_list():

    res = list()
    # load 24h remained feature from csv file
    remained_feature_name = pd.read_csv(
        '/panfs/pfs.local/work/liu/xzhang_sta/tangxizhuo/data/24/24h_snap1_rm1_remain_feature.csv',
        header=None).squeeze("columns").tolist()
    for feature in remained_feature_name:
        if feature.startswith('MED') or feature.startswith('PX'):
            res.append(feature)

    return res


def get_med_px_max_distance():
    load_file_name = '/panfs/pfs.local/work/liu/xzhang_sta/tangxizhuo/data/24/all_24h_snap1_rm1.feather'
    all_sample = pd.read_feather(load_file_name)
    med_px_feature_list = get_med_px_feature_list()
    med_px_all_sample = all_sample.loc[:, med_px_feature_list]
    # get max value of med and px feature
    max_time_distance = med_px_all_sample.max(axis=0)
    max_time_distance = max_time_distance * 2

    return max_time_distance


def fill_miss2_all():
    """---miss 2---
    med and px: 2 * max
    other: 0"""

    # set read file and save file
    load_file_name = '/panfs/pfs.local/work/liu/xzhang_sta/tangxizhuo/data/24/all_24h_snap1_rm1.feather'
    save_file_name = '/panfs/pfs.local/work/liu/xzhang_sta/tangxizhuo/data/24/all_24h_snap1_rm1_miss2.feather'

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


def fill_miss2_each_year():
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
            f"/panfs/pfs.local/work/liu/xzhang_sta/tangxizhuo/data/24/{year}_24h_snap1_rm1_miss2.feather")


def fill_miss3_each_year():
    med_px_feature_list = get_med_px_feature_list()

    # traverse each year and normalize feature
    for year in range(2010, 2018 + 1):
        cur_data = pd.read_feather(
            f"/panfs/pfs.local/work/liu/xzhang_sta/tangxizhuo/data/24/{year}_24h_snap1_rm1_miss1_norm1.feather")
        cur_med_px_data = cur_data.loc[:, med_px_feature_list]
        cur_med_px_data[cur_med_px_data == 0] = np.nan
        cur_data.loc[:, med_px_feature_list] = cur_med_px_data
        # save file
        cur_data.to_feather(
            f"/panfs/pfs.local/work/liu/xzhang_sta/tangxizhuo/data/24/{year}_24h_snap1_rm1_miss3_norm1.feather")


fill_miss3_each_year()

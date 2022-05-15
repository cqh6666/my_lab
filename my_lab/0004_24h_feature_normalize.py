#encoding=gbk
"""
input:
    {year}_24h_dataframe_999_feature_remove.feather
    all_24h_dataframe_999_feature_remove.feather
output:
    all_24h_dataframe_999_feature_normalize.feather
    {year}_24h_dataframe_999_feature_normalize.feather
"""
import pandas as pd

# ---------- work space ----------
# 初始标准化特征列表
normalize_feature_list = ['DEMO_Age', 'VITAL_Height', 'VITAL_Weight', 'VITAL_BMI', 'VITAL_SBP', 'VITAL_DBP']


def collect_specified_features_to_normalize():
    """
    收集所有需要标准化的特征列表
    collect features needed to be normalized: age, height, weight, bmi, sbp, dbp, all lab
    :return:
    """
    remained_feature_name = pd.read_csv(
        '/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/data/24h/24h_999_remained_feature.csv',
        header=None).squeeze("columns").tolist()

    for feature in remained_feature_name:
        if feature.startswith('LAB'):
            normalize_feature_list.append(feature)


def normalize_all_samples():
    """
    对一些特殊特征进行标准化所有样本
    normalize specified features for all samples
    :return:
    """
    all_sample = pd.read_feather('/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/data/24h/all_24h_dataframe_999_feature_remove.feather')

    feature_happen_count = (all_sample.loc[:, normalize_feature_list] != 0).sum(axis=0)
    feature_sum = all_sample.loc[:, normalize_feature_list].sum(axis=0)
    feature_average_if = feature_sum / feature_happen_count
    all_sample.loc[:, normalize_feature_list] = all_sample.loc[:, normalize_feature_list] / feature_average_if

    print("all_sample shape:", all_sample.shape)
    print("test value LAB_521: ", all_sample.loc[80386, 'LAB_521'])
    print("test value other: ", all_sample.loc[80386, 'ID'])
    # save all samples to one file after normalizing feature
    all_sample.to_feather('/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/data/24h/all_24h_dataframe_999_feature_normalize.feather')


def normalize_each_year_samples(years):
    """
    与前面函数不一样，对每一年进行标准化
    years = [2010, 2011, ... , 2018]
    :return:
    """
    # traverse each year and normalize feature
    for year in years:
        cur_data = pd.read_feather(f"/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/data/24h/{year}_24h_dataframe_999_feature_remove.feather")

        feature_happen_count = (cur_data.loc[:, normalize_feature_list] != 0).sum(axis=0)
        feature_sum = cur_data.loc[:, normalize_feature_list].sum(axis=0)
        feature_average_if = feature_sum / feature_happen_count
        cur_data.loc[:, normalize_feature_list] = cur_data.loc[:, normalize_feature_list] / feature_average_if

        cur_data.to_feather(f"/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/data/24h/{year}_24h_dataframe_999_feature_normalize.feather")


if __name__ == '__main__':
    collect_specified_features_to_normalize()
    normalize_all_samples()

    # years = [year for year in range(2010,2018 + 1)]
    # normalize_each_year_samples(years)
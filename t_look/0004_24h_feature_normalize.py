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

# ----- collect features needed to be normalized: age, height, weight, bmi, sbp, dbp, all lab -----
normalize_feature_list = ['DEMO_Age', 'VITAL_Height', 'VITAL_Weight', 'VITAL_BMI', 'VITAL_SBP', 'VITAL_DBP']
# load 24h remained feature from csv file
remained_feature_name = pd.read_csv('/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/data/24h_999_remained_feature.csv',
                                    header=None).squeeze("columns").tolist()
for feature in remained_feature_name:
    if feature.startswith('LAB'):
        normalize_feature_list.append(feature)


# ----- load all samples (after feature removing) -----
all_sample = pd.read_feather('/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/data/24h/all_24h_dataframe_999_feature_remove.feather')


def normalize_all_samples():
    """
    normalize specified features for all samples
    对一些特殊特征进行标准化所有样本
    :return:
    """
    all_sample = pd.read_feather('/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/data/24h/all_24h_dataframe_999_feature_remove.feather')
    feature_happen_count = (all_sample.loc[:, normalize_feature_list] != 0).sum(axis=0)


# optional----- where is the positive elements -----
# print((all_sample < 0).values.sum())
# row = all_sample.index[np.where(all_sample < 0)[0]].tolist()
# column = all_sample.columns[np.where(all_sample < 0)[1]].tolist()
# points = list(zip(row, column))
# for x, y in points:
#     print(f"index-{x}, column-{y}: {all_sample.loc[x, y]}")


# optional----- positive occurrence of LAB_521 -----
# print((all_sample.loc[:, 'LAB_521'] > 0).values.sum())


# ----- normalize specified features for all samples
feature_happen_count = (all_sample.loc[:, normalize_feature_list] != 0).sum(axis=0)
feature_sum = all_sample.loc[:, normalize_feature_list].sum(axis=0)
feature_average_if = feature_sum / feature_happen_count
all_sample.loc[:, normalize_feature_list] = all_sample.loc[:, normalize_feature_list] / feature_average_if
print("all_sample shape:", all_sample.shape)
print("test value LAB_521: ", all_sample.loc[80386, 'LAB_521'])
print("test value other: ", all_sample.loc[80386, 'ID'])
# save all samples to one file after normalizing feature
all_sample.to_feather('/panfs/pfs.local/work/liu/xzhang_sta/tangxizhuo/data/all_24h_dataframe_999_feature_normalize.feather')


# ----- normalize specified features for each year -----
feature_happen_count = (all_sample.loc[:, normalize_feature_list] != 0).sum(axis=0)
feature_sum = all_sample.loc[:, normalize_feature_list].sum(axis=0)
feature_average_if = feature_sum / feature_happen_count
# traverse each year and normalize feature
for year in range(2010, 2018 + 1):
    cur_data = pd.read_feather(f"/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/data/24h/{year}_24h_dataframe_999_feature_remove.feather")
    cur_data.loc[:, normalize_feature_list] = cur_data.loc[:, normalize_feature_list] / feature_average_if
    cur_data.to_feather(f"/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/data/24h/{year}_24h_dataframe_999_feature_normalize.feather")


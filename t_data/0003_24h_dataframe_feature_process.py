"""
input:
    {year}_24h_list2dataframe.feather
output:
    all_24h_list2dataframe.feather
    new_feature_map.csv
    all_24h_dataframe_999_feature_remove.feather
    24h_null_feature.csv
    24h_999_missing_feature.csv
    24h_999_remained_feature.csv
    {year}_24h_dataframe_999_feature_remove.feather
"""
import pandas as pd


def remap_feature_name(samples):
    """copy from BR senior"""
    demoNameTable = {
        "1": "Age", "2": "Hispanic", "3": "Race",
        "4": "Sex"
    }
    vitalNameTable = {
        "1": "Height", "2": "Weight", "3": "BMI",
        "4": "Smoking", "5": "Tobacco", "6": "TobaccoType",
        "7": "SBP", "8": "DBP"
    }
    featureNames = samples.columns
    idNames, demoNames, vitalNames, labNames, medNames, ccsNames, pxNames = [], [], [], [], [], [], []
    for name in featureNames:
        prefix = name[: 2]
        if prefix == "en":
            idNames.append("ID")
            continue
        if prefix == "de":  # 32
            index = name[4]
            newName = "DEMO_" + demoNameTable[index]
            if len(name) > 5:
                value = name[5:]
                newName = newName + "_" + value
            demoNames.append(newName)
        elif prefix == "vi":  # 32
            index = name[5]
            value = name[6:]
            newName = "VITAL_" + vitalNameTable[index]
            if value != "":
                newName = newName + "_" + value
            vitalNames.append(newName)
        elif prefix == "la":  # 817
            index = name[3:]
            newName = "LAB_" + index
            labNames.append(newName)
        elif prefix == "cc":  # 280
            index = name[3:]
            newName = "CCS_" + index
            ccsNames.append(newName)
        elif prefix == "px":  # 15606
            index = name[2:]
            newName = "PX_" + index
            pxNames.append(newName)
        elif prefix == "me":  # 15539
            index = name[3:]
            newName = "MED_" + index
            medNames.append(newName)
    newFeatureName = idNames + demoNames + vitalNames + labNames + ccsNames + pxNames + medNames + ["Label"]
    return newFeatureName


def delete_null_feature(samples):
    """if all values of a feature is zero ,delete it"""
    return samples.loc[:, samples.any()]


def get_high_missing_feature(samples, rate=0.999):
    """if the number of missing values over rate, the corresponding feature will be deleted"""
    high_missing_feature_name = []
    # the upper limit of numbers of missing samples
    max_size = round(samples.shape[0] * rate)
    # 'Label' is not a real feature
    for column in samples.columns[:-1]:
        if (samples[column] == 0).sum() > max_size:
            high_missing_feature_name.append(column)
    return high_missing_feature_name


# ---------- work space ----------

# optional ---------- load dataframe of all years (multiple files) and combine them ----------
# # init a dataframe to load dataframe of all years
# all_sample = pd.DataFrame()
# # traverse all years (2010 ~ 2018) and concat all samples
# for year in range(2010, 2018 + 1):
#     load_df_path = f"/panfs/pfs.local/work/liu/xzhang_sta/tangxizhuo/data/{year}_24h_list2dataframe.feather"
#     all_sample = pd.concat([all_sample, pd.read_feather(load_df_path)], axis=0)
# # reset index of all samples
# all_sample.reset_index(drop=True, inplace=True)


# optional ---------- save all samples to one file ----------
# print("all sample df shape:", all_sample.shape)
# all_sample.to_feather('/panfs/pfs.local/work/liu/xzhang_sta/tangxizhuo/data/all_24h_list2dataframe.feather')


# ---------- load dataframe of all years (one file) ----------
# all_sample = pd.read_feather('/panfs/pfs.local/work/liu/xzhang_sta/tangxizhuo/data/all_24h_list2dataframe.feather')


# optional ---------- delete 'days' and remap feature name (through function) ----------
# # delete 'days'
# all_sample.drop(['days'], axis=1, inplace=True)
# # remap feature name
# all_sample.columns = remap_feature_name(all_sample)


# optional ---------- save new feature name (without 'days' compared to previous feature names) ----------
# pd.Series(all_sample.columns).to_csv("/panfs/pfs.local/work/liu/xzhang_sta/tangxizhuo/data/new_feature_map.csv",
#                                      index=False,
#                                      header=False)


# ----- get new feature map name through csv file -----
new_feature_name = pd.read_csv('/panfs/pfs.local/work/liu/xzhang_sta/tangxizhuo/data/new_feature_map.csv',
                               header=None).squeeze("columns")

# ---------- for all_sample, delete 'days' and remap feature name (through csv file) ----------
# # delete 'days'
# all_sample.drop(['days'], axis=1, inplace=True)
# # remap feature name, new feature name -> pd.Series
# all_sample.columns = new_feature_name


# optional ----- delete null feature (through function) and save deleted feature name as csv-----
# all_sample = delete_null_feature(all_sample)
# # get deleted null feature names
# deleted_null_feature_name = list(set(new_feature_name) - set(all_sample.columns))
# # save null feature name to csv
# pd.Series(deleted_null_feature_name).to_csv('/panfs/pfs.local/work/liu/xzhang_sta/tangxizhuo/data/24h_null_feature.csv',
#                                             index=False,
#                                             header=False)


# ---------- delete null feature (through csv file) ----------
# deleted_null_feature_name = pd.read_csv('/panfs/pfs.local/work/liu/xzhang_sta/tangxizhuo/data/24h_null_feature.csv', header=None).squeeze("columns")
# all_sample.drop(deleted_null_feature_name, axis=1, inplace=True)


# optional----- delete high missing (through function) after delete null feature, and save feature name as csv -----
# deleted_high_missing_feature_name = get_high_missing_feature(all_sample, rate=0.999)
# # save high missing feature name to csv
# pd.Series(deleted_high_missing_feature_name).to_csv('/panfs/pfs.local/work/liu/xzhang_sta/tangxizhuo/data/24h_999_missing_feature.csv',
#                                                     index=False,
#                                                     header=False)


# ----- delete high missing feature (through csv file) after deleting null feature -----
# deleted_high_missing_feature_name = pd.read_csv('/panfs/pfs.local/work/liu/xzhang_sta/tangxizhuo/data/24h_999_missing_feature.csv', header=None).squeeze("columns")
# all_sample.drop(deleted_high_missing_feature_name, axis=1, inplace=True)


# optional----- get remained feature name through all_sample-----
# remained_feature_name = all_sample.columns


# optional----- save remained feature as csv -----
# pd.Series(remained_feature_name).to_csv('/panfs/pfs.local/work/liu/xzhang_sta/tangxizhuo/data/24h_999_remained_feature.csv',
#                                         index=False,
#                                         header=False)


# ----- get remained feature through csv file -----
remained_feature_name = pd.read_csv('/panfs/pfs.local/work/liu/xzhang_sta/tangxizhuo/data/24h_999_remained_feature.csv',
                                    header=None).squeeze("columns")

# optional----- combine all years after feature processing -----
# all_sample = all_sample.loc[:, remained_feature_name]
# all_sample.to_feather('/panfs/pfs.local/work/liu/xzhang_sta/tangxizhuo/data/all_24h_dataframe_999_feature_remove.feather')


# optional----- for every year samples, retain remained features and save them as feather -----
for year in range(2010, 2018 + 1):
    origin = pd.read_feather(f"/panfs/pfs.local/work/liu/xzhang_sta/tangxizhuo/data/{year}_24h_list2dataframe.feather")
    origin.drop(['days'], axis=1, inplace=True)
    origin.columns = new_feature_name
    new = origin.loc[:, remained_feature_name]
    print(f"-----year {year}-----")
    print("shape is :", new.shape)
    new.to_feather(
        f"/panfs/pfs.local/work/liu/xzhang_sta/tangxizhuo/data/{year}_24h_dataframe_999_feature_remove.feather")

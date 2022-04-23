#encoding=gbk
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
from multiprocessing import Pool
import time
from my_logger import MyLog
import os

my_logger = MyLog().logger

feather_file = '/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/data/24h/all_24h_dataframe_999_feature_remove.feather'
csv_file = '/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/data/24h/all_24h_dataframe_999_feature_remove.csv'
pickle_file = '/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/data/24h/all_24h_dataframe_999_feature_remove.pkl'

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
# ----- get new feature map name through csv file -----
new_feature_name = pd.read_csv('/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/data/new_feature_map.csv',
                               header=None).squeeze("columns")

# ----- get remained feature through csv file -----
# 读取经过缺失筛选后的特征集合
remained_feature_name = pd.read_csv('/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/data/24h/24h_999_remained_feature.csv',
                                    header=None).squeeze("columns")


def run(year):
    origin = pd.read_feather(
        f"/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/data/24h/{year}_24h_list2dataframe.feather")
    origin.drop(['days'], axis=1, inplace=True)
    origin.columns = new_feature_name
    new = origin.loc[:, remained_feature_name]
    print(f"---------- year {year} ---------")
    print("shape is :", new.shape)
    # 保存路径
    new.to_feather(
        f"/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/data/24h/{year}_24h_dataframe_999_feature_remove.feather")


def collect_all_samples():
    """
    将从2010-2018年的数据全部收集成为一个文件
    :return:
    """
    all_sample = pd.DataFrame()
    for year in range(2010, 2018 + 1):
        load_df_path = f"/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/data/24h/{year}_24h_dataframe_999_feature_remove.feather"
        all_sample = pd.concat([all_sample, pd.read_feather(load_df_path)], axis=0)
        all_sample.reset_index(drop=True, inplace=True)

    print("all sample df shape:", all_sample.shape)

    start_time = time.time()
    all_sample.to_feather(feather_file)
    feather_time = time.time()

    # save as csv
    all_sample.to_csv(csv_file)
    csv_time = time.time()

    all_sample.to_pickle(pickle_file)
    pickle_time = time.time()

    my_logger.info(f"save feather time: {feather_time - start_time}")
    my_logger.info(f"save csv time: {csv_time - feather_time}")
    my_logger.info(f"save pickle time: {pickle_time - csv_time}")


def compare_file_type():
    feather_size = hum_convert(os.path.getsize(feather_file))
    csv_size = hum_convert(os.path.getsize(csv_file))
    pickle_size = hum_convert(os.path.getsize(pickle_file))

    my_logger.info(f"feather size: {feather_size}")
    my_logger.info(f"csv size: {csv_size}")
    my_logger.info(f"pickle size: {pickle_size}")

    start_time = time.time()
    pd.read_feather(feather_file)
    feather_time = time.time()
    pd.read_csv(csv_file)
    csv_time = time.time()
    pd.read_pickle(pickle_file)
    pickle_time = time.time()

    my_logger.info(f"load feather time: {feather_time - start_time}")
    my_logger.info(f"load csv time: {csv_time - feather_time}")
    my_logger.info(f"load pickle time: {pickle_time - csv_time}")


def hum_convert(value):
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    size = 1024.0
    for i in range(len(units)):
        if (value / size) < 1:
            return "%.2f%s" % (value, units[i])
        value = value / size


if __name__ == '__main__':
    # cpu_worker_num = 20
    # process_year = [year for year in range(2010, 2018+1)]
    # start_time = time.time()
    # with Pool(cpu_worker_num) as p:
    #     outputs = p.map(run, process_year)
    # print(f'| outputs: {outputs}    TimeUsed: {time.time() - start_time:.1f}    \n')

    # 整合
    collect_all_samples()
    compare_file_type()
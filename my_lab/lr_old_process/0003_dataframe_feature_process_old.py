# encoding=gbk
"""
ֻ�Ƕ�����������������999ɸѡ������ȥ����days��ǩ������ϸ��û�н��д���
input:
    {year}_24h_list2dataframe.feather
output:
    all_{pre_hour}h_df_rm1.feather
    new_feature_map.csv
    24h_999_remained_feature_map.csv
    {year}_{pre_hour}h_df_rm1.feather
"""
import pandas as pd
import time
from my_logger import MyLog
import os


def remap_feature_name(old_feature_list, is_save=True):
    """
    ������ӳ�䣬������csv�ļ�
    :param old_feature_list: �ɵ������б�
    :param is_save: �Ƿ񱣴�
    :return:
    """
    demoNameTable = {
        "1": "Age", "2": "Hispanic", "3": "Race",
        "4": "Sex"
    }
    vitalNameTable = {
        "1": "Height", "2": "Weight", "3": "BMI",
        "4": "Smoking", "5": "Tobacco", "6": "TobaccoType",
        "7": "SBP", "8": "DBP"
    }
    featureNames = old_feature_list
    idNames, demoNames, vitalNames, labNames, medNames, ccsNames, pxNames = [], [], [], [], [], [], []
    for name in featureNames:
        prefix = name[: 2]
        # encounter_id => ID [0]
        if prefix == "en":
            idNames.append("ID")
            continue
        # demo3NI => DEMO_Race_NI [1,23]
        elif prefix == "de":
            index = name[4]
            newName = "DEMO_" + demoNameTable[index]
            if len(name) > 5:
                value = name[5:]
                newName = newName + "_" + value
            demoNames.append(newName)
        # vital401 => VITAL_Smoking_01 [24,55]
        elif prefix == "vi":
            index = name[5]
            value = name[6:]
            newName = "VITAL_" + vitalNameTable[index]
            if value != "":
                newName = newName + "_" + value
            vitalNames.append(newName)
        # lab10 => LAB_10 [56,872] 817
        elif prefix == "la":
            index = name[3:]
            newName = "LAB_" + index
            labNames.append(newName)
        # ccs10 => CCS_10 [873,1152]  280
        elif prefix == "cc":
            index = name[3:]
            newName = "CCS_" + index
            ccsNames.append(newName)
        # PX10 => px_10  [1153,1153 + 15606]
        elif prefix == "px":  #
            index = name[2:]
            newName = "PX_" + index
            pxNames.append(newName)
        # MED10006 => MED_10006
        elif prefix == "me":  # 15539
            index = name[3:]
            newName = "MED_" + index
            medNames.append(newName)
    newFeatureName = idNames + demoNames + vitalNames + labNames + ccsNames + pxNames + medNames + ["Label"]
    my_logger.info(f"remap_feature_name | len:{len(newFeatureName)} | ID index:0 | Label index:{len(newFeatureName)-1}")

    if is_save:
        new_feature_name_df = pd.DataFrame({'new_feature': newFeatureName})
        new_feature_name_df.to_csv(os.path.join(DATA_SOURCE_PATH, "new_feature_map.csv"), index=False, header=0)

        # �����¾�����ӳ��
        old_and_new_feature_name_df = pd.DataFrame({'old_feature': featureNames, 'new_feature': newFeatureName})
        old_and_new_feature_name_df.to_csv(os.path.join(DATA_SOURCE_PATH, "old_and_new_feature_map.csv"), index=False)
        my_logger.info("save old and new feature map csv...")

    return newFeatureName


def delete_null_feature(samples):
    """
    ȥ��ֵ���������ֵ��
    any() ���ȫΪ�ջ�0��false���򷵻�false�������true��
    :param samples: ����
    :return:
    """
    return samples.loc[:, samples.any()]


def get_high_missing_feature(samples, rate=0.999):
    """
    if the number of missing values over rate, the corresponding feature will be deleted
    ��ȡȱʧֵ���������
    :param samples:
    :param rate: ȱʧ��
    :return:
    """
    high_missing_feature_name = []
    # the upper limit of numbers of missing samples
    max_size = round(samples.shape[0] * rate)
    # 'Label' is not a real feature
    for column in samples.columns[:-1]:
        if (samples[column] == 0).sum() > max_size:
            high_missing_feature_name.append(column)
    return high_missing_feature_name


def compare_high_rate_missing_feature(samples, rate_list=None):
    """
    �Ƚϲ�ͬȱʧ��ɸѡ���������
    :param samples:
    :param rate_list:
    :return:
    """
    # set default list
    if rate_list is None:
        rate_list = [0.999, 0.99, 0.95, 0.90]

    missing_feature_nums = []
    for rate in rate_list:
        len_feature = len(get_high_missing_feature(samples, rate))
        missing_feature_nums.append(len_feature)

    my_logger.info(f"rate_list:{rate_list} | missing_feature_nums:{missing_feature_nums}")
    return missing_feature_nums


def collect_all_samples(file_name="list2dataframe.feather", start_year=2010, end_year=2018):
    """
    ��������ݵ�����ȫ���ռ���Ϊһ���ļ�
    :param file_name: �ļ���
    :param pre_hour: ��ǰСʱ
    :param start_year: ��ʼ���
    :param end_year: �������
    :return: all_samples
    """
    assert (start_year < end_year)

    save_file_name = os.path.join(DATA_SOURCE_PATH, f"{pre_hour}h_all_{file_name}")
    if os.path.exists(save_file_name):
        my_logger.warning(f"exist {save_file_name}, will not collect all samples...")
        return pd.read_feather(save_file_name)

    all_sample = pd.DataFrame()
    for year in range(start_year, end_year + 1):
        try:
            load_df_path = os.path.join(DATA_SOURCE_PATH, f"{year}_{pre_hour}h_{file_name}")
            my_logger.info(f"load source samples... - [{load_df_path}]")
            all_sample = pd.concat([all_sample, pd.read_feather(load_df_path)], axis=0)
            all_sample.reset_index(drop=True, inplace=True)
        except Exception as err:
            my_logger.error(err)

    my_logger.info(f"all sample of shape from {start_year} to {end_year}: {all_sample.shape}")

    all_sample.to_feather(save_file_name)
    my_logger.info(f"save all samples to feather success! - [{save_file_name}]")
    return all_sample


def get_dataframe_by_year(file_name="list2dataframe.feather", pre_hour=24, start_year=2010, end_year=2018):
    """
    ���ÿ����ݵ�dataframe������feature_list��ɸѡ����
    :param end_year:
    :param start_year:
    :param file_name:
    :param pre_hour: Сʱ
    :return:
    """
    load_first_path = os.path.join(DATA_SOURCE_PATH, f"{start_year}_{pre_hour}h_df_rm1.feather")
    if os.path.exists(load_first_path):
        my_logger.warning(f"exist {load_first_path}, will not get dataframe in every year...")
        return

    new_feature_file = os.path.join(DATA_SOURCE_PATH, "new_feature_map.csv")
    new_feature_map = pd.read_csv(new_feature_file, header=None).squeeze().tolist()
    remained_feature_name_path = os.path.join(DATA_SOURCE_PATH, f"{pre_hour}_999_remained_new_feature_map.csv")
    remained_feature_map = pd.read_csv(remained_feature_name_path, header=None).squeeze().tolist()

    for year in range(start_year, end_year + 1):
        load_df_path = os.path.join(DATA_SOURCE_PATH, f"{year}_{pre_hour}h_{file_name}")
        my_logger.warning(f"load source samples... - [{load_df_path}]")
        source_sample = pd.read_feather(load_df_path)
        source_sample.drop(['days'], axis=1, inplace=True)
        source_sample.columns = new_feature_map

        remained_source_sample = source_sample.loc[:, remained_feature_map]
        save_file_name = os.path.join(DATA_SOURCE_PATH, f"{year}_{pre_hour}h_df_rm1.feather")
        remained_source_sample.to_feather(save_file_name)
        my_logger.info(f"save {year} samples to feather success... shape: {remained_source_sample.shape} | [{save_file_name}]")


def run(start_year=2010, end_year=2018):
    """
    ����ڣ�ǰ��������list2dataframe���ݼ�
    :param start_year: ��ʼ���
    :param end_year: �������
    :param pre_hour: 24h / 48h /72h
    :return:
    """
    # load data
    all_samples = collect_all_samples(file_name="list2dataframe.feather", start_year=start_year, end_year=end_year)

    # drop days lab817
    all_samples.drop(['days', 'lab817'], axis=1, inplace=True)

    # feature process map
    old_feature_map = all_samples.columns
    new_feature_name = remap_feature_name(old_feature_map)  # list
    all_samples.columns = new_feature_name

    # delete null feature
    all_samples = delete_null_feature(all_samples)

    my_logger.warning("=============JUST TEST ===============")
    # �Ƚ��²�ͬȱʧ�ʶ�Ӧȱʧ����������
    compare_high_rate_missing_feature(all_samples)
    my_logger.warning(f"============== END ==================")

    # delete 999 feature
    high_missing_feature = get_high_missing_feature(all_samples)
    all_samples.drop(high_missing_feature, axis=1, inplace=True)

    # remained feature and save
    remained_feature_name = all_samples.columns
    remained_feature_name_df = pd.DataFrame({"remained_feature": remained_feature_name})
    remained_feature_name_path = os.path.join(DATA_SOURCE_PATH, f"remained_new_feature_map.csv")
    remained_feature_name_df.to_csv(remained_feature_name_path, index=False, header=0)
    my_logger.info(f"999_remained_feature len:{len(remained_feature_name)}, save to [{remained_feature_name_path}]")

    # save all data
    save_file = os.path.join(DATA_SOURCE_PATH, f"all_{pre_hour}h_df_rm1.feather")
    all_samples.to_feather(save_file)
    my_logger.info(f"save all_samples to feather - [{save_file}]")


if __name__ == '__main__':
    pre_hour = 24
    # csv_file = '/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/data/24h/all_24h_dataframe_999_feature_remove.csv'
    # pickle_file = '/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/data/24h/all_24h_dataframe_999_feature_remove.pkl'
    root_dir = f"{pre_hour}h_old2"
    DATA_SOURCE_PATH = f"/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/data/{root_dir}/"
    my_logger = MyLog().logger

    run()
    # # �õ�����ÿһ������ݣ���ʱ���õ�
    # get_dataframe_by_year(pre_hour=24)

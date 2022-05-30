"""
input:
    yuanborong/data/row_data_encounterId/{year}_string2list.pkl
    yuanborong/data/row_data/feature_dict_BDAI_map.pkl
output:
    {year}_24h_list2dataframe.feather

transform nested list to cross-section data according to pre_day (pre_day=1 here)
no feature filtration
use multi-process and shared-memory

#2022-4-19

"""
import joblib
import numpy as np
import pandas as pd
from multiprocessing import Pool, shared_memory
from my_logger import MyLog


def get_encounter_id(idx, input_data, map_data, all_sample):
    encounter_id_index = map_data.index("encounter_id")
    all_sample[idx, encounter_id_index] = input_data


def get_demo(idx, input_data, map_data, all_sample):
    Demo = np.array(input_data)
    # Gets the eigenvalue corresponding to the subscript index
    for m in range(len(Demo)):
        demo_index = "demo"
        if m == 0:
            demo_index = demo_index + str(1)
        else:
            demo_index = demo_index + str(m + 1) + str(Demo[m])
            Demo[m] = 1

        indexNum = map_data.index(demo_index)
        all_sample[idx, indexNum] = Demo[m]


def get_vital(idx, input_data, input_t_time, map_data, all_sample):
    for m in range(len(input_data)):
        try:
            temp = input_data[m]
            temp1 = np.asarray(temp)
            temp2 = temp1[:, -1]
            temp2 = list(map(float, temp2))
            temp3 = [x for x in temp2 if x <= input_t_time]
            temp4 = np.max(temp3)
            temp2.reverse()
            temp5 = temp2.index(temp4)
            temp5 = len(temp2) - 1 - temp5

            if m == 3 or m == 4 or m == 5:
                # vital4,vital5,vital6 is a qualitative variable
                # Gets the eigenvalue corresponding to the subscript index
                vitalIndex = 'vital' + str((m + 1) * 10) + str(int(temp[temp5][0]))
                indexNum = map_data.index(vitalIndex)
                all_sample[idx, indexNum] = 1
                continue
            else:
                # Gets the eigenvalue corresponding to the subscript index
                vitalIndex = 'vital' + str(m + 1)
                indexNum = map_data.index(vitalIndex)
                all_sample[idx, indexNum] = temp[temp5][0]
        except:
            continue


def get_lab(idx, input_data, input_t_time, map_data, all_sample):
    for m in range(len(input_data)):
        try:
            labIndex = input_data[m][0][0][0]
            # Gets the eigenvalue corresponding to the subscript index
            indexNum = map_data.index(labIndex)
            temp = input_data[m][1]
            temp1 = np.asarray(temp)
            temp2 = temp1[:, -1]
            temp2 = list(map(float, temp2))
            temp3 = [x for x in temp2 if x <= input_t_time]
            temp4 = np.max(temp3)
            temp2.reverse()
            temp5 = temp2.index(temp4)
            temp5 = len(temp2) - 1 - temp5
            all_sample[idx, indexNum] = temp[temp5][0]
        except:
            continue
    pass


def get_med(idx, input_data, input_t_time, pre_day, map_data, all_sample):
    """
    we set days to AKI point distances as med values if non-exposure , we set NaN
    """
    for m in range(len(input_data)):
        try:
            med_index = input_data[m][0][0][0]
            # Gets the eigenvalue corresponding to the subscript index
            indexNum = map_data.index(med_index)
            temp = input_data[m][1]
            temp1 = np.asarray(temp)
            temp2 = temp1[:, -1]
            temp2 = list(map(float, temp2))
            temp3 = [x for x in temp2 if x <= input_t_time]
            temp4 = np.max(temp3)
            # we set days to AKI point distances as med values
            # if AKI is day 6 , pre_day is 1 .
            # so input_t_time = 6 - 1 = 5 (predict point)
            # temp4 is nearest day time(for example , temp4 = 4)
            all_sample[idx, indexNum] = (input_t_time + pre_day) - temp4
        except:
            continue
    pass


def get_ccs(idx, input_data, input_t_time, map_data, all_sample):
    for m in range(len(input_data)):
        try:
            ccsTimes = input_data[m][1]
            ccsTimes = list(map(float, ccsTimes))
            ccsTime = np.min(ccsTimes)
            if ccsTime <= input_t_time:
                # Gets the eigenvalue corresponding to the subscript index
                ccsIndex = input_data[m][0][0]
                indexNum = map_data.index(ccsIndex)
                all_sample[idx][indexNum] = 1
        except:
            continue
    pass


def get_px(idx, input_data, input_t_time, pre_day, map_data, all_sample):
    """
    we set days to AKI point distances as px values if non-procedure , we set NaN
    """
    for m in range(len(input_data)):
        try:
            # Gets the eigenvalue corresponding to the subscript index
            px_index = input_data[m][0][0]
            index_num = map_data.index(px_index)
            px_times = input_data[m][1]
            px_times = list(map(float, px_times))
            px_times_before_pre = [x for x in px_times if x <= input_t_time]
            px_time = np.max(px_times_before_pre)
            all_sample[idx][index_num] = (input_t_time + pre_day) - px_time
        except:
            continue
    pass


def get_label(idx, labels, advance_day, map_data, all_sample):
    """
    Gets the eigenvalue corresponding to the subscript index
    """
    day_index = map_data.index("days")
    value_index = map_data.index("AKI_label")
    for AKI_data in labels:
        if float(AKI_data[0]) > 0:  # 正例样本
            all_sample[idx, day_index] = float(AKI_data[1]) - advance_day  # 提前预测的天数
            all_sample[idx, value_index] = 1  # 预测分类
            break
        else:
            all_sample[idx, day_index] = float(AKI_data[1]) - advance_day
            all_sample[idx, value_index] = 0
    return all_sample[idx, value_index], all_sample[idx, day_index]


def process_cur_sample(idx, cur_list_data, pre_day, map_data, shape_all_sample):
    """
    use the shared_memory for all_sample
    :param idx: 每一行索引
    :param cur_list_data: 每一行所有数据
    :param pre_day: 提前天数（1->24h,2->48h）
    :param map_data: 特征列表
    :param shape_all_sample: all_sample的shape (rows,columns)
    :return:
    """
    # 通过name读共享内存变量
    shared_all_sample = shared_memory.SharedMemory(name='all_sample')
    # 写共享内存变量
    all_sample = np.ndarray(shape=shape_all_sample, dtype=np.float64, buffer=shared_all_sample.buf)

    # get patient_id, aki_label and six kinds of features by nested list
    encounter_id, demo, vital, lab, ccs, px, med, label = cur_list_data

    # get aki_status and specific prediction time according to pre_day and aki_label
    aki_status, pre_time = get_label(idx, label, pre_day, map_data, all_sample)
    if pre_time < 0:
        return idx
    get_encounter_id(idx, encounter_id, map_data, all_sample)
    get_demo(idx, demo, map_data, all_sample)
    get_vital(idx, vital, pre_time, map_data, all_sample)
    get_lab(idx, lab, pre_time, map_data, all_sample)
    get_med(idx, med, pre_time, pre_day, map_data, all_sample)
    get_ccs(idx, ccs, pre_time, map_data, all_sample)
    get_px(idx, px, pre_time, pre_day, map_data, all_sample)

    shared_all_sample.close()


def get_discard_index(x):
    """
    去除无用信息 pre_time < 0
    """
    if x is not None:
        throw_idx.append(x)


# ----------------------------------------- work space ------------------------------------------------------
# how many days in advance to predict
pre_day = 1
# 提前小时
pre_hour = pre_day * 24

start_year = 2010
end_year = 2018

my_logger = MyLog().logger

# load map data: list of feature name 获取特征名
map_file_path = "/home/xzhang_sta/work/yuanborong/data/row_data/feature_dict_BDAI_map.pkl"
s_map_data = joblib.load(map_file_path)
# 增加 病人ID 一列
s_map_data.insert(0, "encounter_id")
# 保存为csv文件 feature_dict_BDAI_map.pkl -> old_feature_map.csv
feature_map_file = "/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/data/old_feature_map.csv"
old_feature_map = pd.DataFrame(data=s_map_data)
old_feature_map.to_csv(feature_map_file)
my_logger.info(f"save feature map csv to - [{feature_map_file}]")

# 特征总数量
len_s_map_data = len(s_map_data)

# traverse data of all years 所有年份的数据
for year in range(start_year, end_year + 1):
    # init the total of samples whose stay was less than 1 day
    throw_idx = []
    # set the input file and output file 输入是list格式，输出是dataFrame格式
    string_list_file_path = f"/panfs/pfs.local/work/liu/xzhang_sta/yuanborong/data/row_data_encounterId/{year}_string2list.pkl"
    save_file_path = f"/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/data/24h/{year}_{pre_hour}h_list2dataframe.feather"

    # 读取list格式的数据
    list_data = joblib.load(string_list_file_path)

    # indexes of samples needed
    len_list_data = len(list_data)
    my_logger.info(f"year:{year} | len_list_data:{len_list_data}")

    # init a variable to save all eligible cross-section sample 初始化数据集格式
    all_sample = np.zeros((len_list_data, len_s_map_data))

    # create shared_memory for all_sample 创建共享内存变量 all_sample
    sm_all_sample = shared_memory.SharedMemory(name='all_sample', create=True, size=all_sample.nbytes)
    s_all_sample = np.ndarray(shape=all_sample.shape, dtype=np.float64, buffer=sm_all_sample.buf)
    s_all_sample[:, :] = all_sample[:, :]
    shape_all_sample = all_sample.shape
    del all_sample

    # init process pool
    pool = Pool(processes=20)

    # traverse all samples in nested list and process them
    for i in range(len_list_data):
        pool.apply_async(func=process_cur_sample,
                         args=(i, list_data[i], pre_day, s_map_data, shape_all_sample),
                         callback=get_discard_index)

    pool.close()
    pool.join()

    # init result (have not abandon < 24h samples)  结果转化为dataFrame
    result = pd.DataFrame(s_all_sample, columns=s_map_data)
    my_logger.info(f"year:{year} | result_shape_before_drop:{result.shape}")
    # discard samples and transform numpy to dataframe
    result.drop(result.index[throw_idx], axis=0, inplace=True)
    # reset index
    result.reset_index(inplace=True, drop=True)

    my_logger.info(f"year:{year} | drop_numbers:{len(throw_idx)}")
    my_logger.info(f"year:{year} | result_shape_after_drop:{result.shape}")

    # save cross-section dataframe as feather file
    result.to_feather(save_file_path)
    my_logger.info(f"save dataframe to feather - [{save_file_path}]")

    sm_all_sample.close()
    sm_all_sample.unlink()

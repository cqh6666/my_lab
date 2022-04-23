# -*- coding: utf-8 -*-
"""
This is the code that handles the AKI_1_2_3 task, with the predicted time being 48 hours ahead.
The program requires four parameters:
parameters 1: The name of the data center to be processed(eg:KUMC) 数据中心
parameters 2: Which year of data from the current data center needs to be processed(eg:2010) 年份
parameters 3: Number of days predicted in advance (1, represents one day in advance) 提前多少天预测

output: multiple "data.csv" files 输出csv文件

Example:
python P3_1_AKI_1_2_3_process.py  KUMC  2010 2
"""
# 生成新的数据集

import numpy as np
import sys
import os
import joblib
from sklearn.model_selection import train_test_split
import functionUtils as fu


# label
def get_label(labels, advance_day):
    # The status and predicted time of AKI in patients with the first outbreak of AKI
    status = int(labels[0][0])
    day = int(labels[0][1])
    # Each patient was tracked from admission until an outbreak of AKI was predicted,
    # and those without AKI were tracked until day 7.
    if status:
        day = day - advance_day
    else:
        day = day if day <= 7 else 7
    return [status, day]


if __name__ == "__main__":
    # site_name = str(sys.argv[1])
    year = str(sys.argv[1])
    pre_day = int(sys.argv[2])
    print("task = AKI_1_2_3  site_name=", "year=", year, "pre_day=", pre_day)
    parent_path = '/panfs/pfs.local/work/liu/xzhang_sta/yaorunze/data/'
    # string_list_file_path为txt转为list后的数据
    string_list_file_path = parent_path + "/list" + "/" + year + '_string2list.pkl'
    # map_file_path为map映射的特征名称文件
    map_file_path = parent_path + "/feature/feature_dict_map.pkl"
    # feature_num_path为各类特征数量文件
    feature_num_path = parent_path + "/feature/feature_num.pkl"
    # 保存的文件路径
    save_file_path = parent_path + "/data/AKI_1_2_3/" + year + "/" + str(pre_day * 24) + "h"

    data = joblib.load(string_list_file_path)  # list n个病历
    map_data = joblib.load(map_file_path)  # dict
    list_num = joblib.load(feature_num_path)  # list
    # column len: 28305
    column_len = len(map_data)
    data_list = fu.data_list
    data_num = len(data)
    no_rolling_data = np.zeros([data_num, column_len], dtype=np.float32)
    # processing encounter one by one
    for i in range(data_num):
        encounter_label = 0
        encounter_id, demo, vital, lab, ccs, px, med, label = data[i]
        # 获取AKI类型和提前多少天
        AKI_status, pred_day = get_label(label, pre_day)

        # Instance with hospital stay less than pred_day time are discarded
        if pred_day < 0:
            continue

        # 28302 28303 28304
        day_index, aki_label_index, id_index = map_data["days"], map_data["AKI_label"], map_data["encounter_id"]

        # if only AKI_status >=2 is recorded, only a positive instance is generated.
        if AKI_status == 2 or AKI_status == 3:
            instance = fu.get_instance(list_num)  # 一行属性， [1，column_len]（column_len属性长度）
            instance[0, day_index] = pred_day
            instance[0, aki_label_index] = 1
            instance[0, id_index] = encounter_id
            encounter_label = 1

            instance = fu.get_demo(demo, map_data, instance)
            instance = fu.get_vital(vital, pred_day, map_data, instance)
            instance = fu.get_lab(lab, pred_day, map_data, instance)
            instance = fu.get_med(med, pred_day, map_data, instance)
            instance = fu.get_ccs(ccs, pred_day, map_data, instance)
            instance = fu.get_px(px, pred_day, map_data, instance)

            # 第一列是病人ID，第二列是病人标签，后面是实例数据
            data_list[i] = np.asarray([encounter_id, encounter_label, instance])
            no_rolling_data[i] = instance
            continue

        # If AKI_status 1 is recorded, a positive instance (a day before pred_day) will be generated,
        # and before the positive instance is all negative instances.
        instances_list = np.zeros([pred_day + 1, column_len])
        for pred_day_sub in range(pred_day + 1):
            # Initialize each instance to a zero matrix
            instance = fu.get_instance(list_num)
            instance[0, day_index] = pred_day_sub
            instance[0, id_index] = encounter_id

            if pred_day_sub == pred_day and AKI_status != 0:
                instance[0, aki_label_index] = 1
                encounter_label = 1
            else:
                instance[0, aki_label_index] = 0

            instance = fu.get_demo(demo, map_data, instance)
            instance = fu.get_vital(vital, pred_day_sub, map_data, instance)
            instance = fu.get_lab(lab, pred_day_sub, map_data, instance)
            instance = fu.get_med(med, pred_day_sub, map_data, instance)
            instance = fu.get_ccs(ccs, pred_day_sub, map_data, instance)
            instance = fu.get_px(px, pred_day_sub, map_data, instance)

            instances_list[pred_day_sub] = instance

        data_list[i] = np.asarray([encounter_id, encounter_label, instances_list])
        no_rolling_data[i] = instances_list[-1]

    #  Release variable
    encounter_id, demo, vital, lab, ccs, px, med, label, instance, data, map_data, instances_list = [], [], [], [], [], [], [], [], [], [], [], []
    del encounter_id, demo, vital, lab, ccs, px, med, label, instance, data, map_data, instances_list
    # save no rolling data
    no_rolling_data = no_rolling_data[no_rolling_data[:, 0] != 0]
    no_rolling_data_df = fu.get_DataFrame(no_rolling_data, list_num)
    no_rolling_data_save_path = save_file_path + "/no_rolling/"
    isExists = os.path.exists(no_rolling_data_save_path)
    if not isExists:
        os.makedirs(no_rolling_data_save_path)
    no_rolling_data_df.to_csv(no_rolling_data_save_path + '/data.csv', index=False)

    # 新增修改
    no_rolling_data_train, no_rolling_data_test, _, _ = train_test_split(no_rolling_data_df,
                                                                         no_rolling_data_df.iloc[:, 1], test_size=0.3,
                                                                         random_state=42,
                                                                         stratify=no_rolling_data_df.iloc[:, 1])
    no_rolling_data_train.to_csv(no_rolling_data_save_path + '/train_data.csv', index=False)
    no_rolling_data_test.to_csv(no_rolling_data_save_path + '/test_data.csv', index=False)

    print("no rolling data size:", no_rolling_data_df.shape)
    print("labels:")
    print(no_rolling_data_df.iloc[:, -2].value_counts())
    print("no rolling train data size:", no_rolling_data_train.shape)
    print("labels:")
    print(no_rolling_data_train.iloc[:, -2].value_counts())
    print("no rolling test data size:", no_rolling_data_test.shape)
    print("labels:")
    print(no_rolling_data_test.iloc[:, -2].value_counts())

    del no_rolling_data, no_rolling_data_df, no_rolling_data_train, no_rolling_data_test

    # Data = np.asarray([x for x in data_list if x != 0])
    # data_list = []

    # Divide training and test set
    # Note that: divide dataset according to the encounters' ID
    # train_data, t_v_data, _, t_v_label = train_test_split(Data, Data[:, 1], test_size=0.3, random_state=42,
    #                                                      stratify=Data[:, 1])
    # Data = []

    # Divide test and calibration set
    # if year in ["2017", "2018"]:
    #    save_file_path = save_file_path + "/temporal"

    # fu.saveData(column_len, train_data, save_file_path + "/train/", list_num)
    # train_data, _ = [], []

    # test_data, calibration_data, _, _ = train_test_split(t_v_data, t_v_label, test_size=0.5, random_state=42,
    #                                                    stratify=t_v_label)

    # fu.saveData(column_len, test_data, save_file_path + "/test/", list_num)
    # fu.saveData(column_len, calibration_data, save_file_path + "/calibration/", list_num)

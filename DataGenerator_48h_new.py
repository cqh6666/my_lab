# -*- coding: utf-8 -*-
"""
This code is used to generate data sets for the day before the onset of illness
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd

# demo
def get_demo(input_data):
    Demo = np.array(input_data)
    # Gets the eigenvalue corresponding to the subscript index
    for m in range(len(Demo)):
        demo_index = "demo"
        if m == 0:
            demo_index = demo_index + str(1)
        else:
            demo_index = demo_index + str(m + 1) + str(Demo[m])
            Demo[m] = 1

        indexNum = list(map_data).index(demo_index)
        valueAll[0, indexNum] = Demo[m]

# vital
def get_vital(input_data, input_t_time):
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
                indexNum = list(map_data).index(vitalIndex)
                valueAll[0, indexNum] = 1
                continue
            else:
                # Gets the eigenvalue corresponding to the subscript index
                vitalIndex = 'vital' + str(m + 1)
                indexNum = list(map_data).index(vitalIndex)
                valueAll[0, indexNum] = temp[temp5][0]
        except:
            # if it's continuous variables and it's missing , we fill it with nan
            if m != 3 and m != 4 and m != 5:
                vitalIndex = 'vital' + str(m + 1)
                indexNum = list(map_data).index(vitalIndex)
                valueAll[0, indexNum] = None
            continue

# lab
def get_lab(input_data, input_t_time):
    for m in range(len(input_data)):
        try:
            labIndex = input_data[m][0][0][0]
            # Gets the eigenvalue corresponding to the subscript index
            indexNum = list(map_data).index(labIndex)
            temp = input_data[m][1]
            temp1 = np.asarray(temp)
            temp2 = temp1[:, -1]
            temp2 = list(map(float, temp2))
            temp3 = [x for x in temp2 if x <= input_t_time]
            temp4 = np.max(temp3)
            temp2.reverse()
            temp5 = temp2.index(temp4)
            temp5 = len(temp2) - 1 - temp5
            valueAll[0, indexNum] = temp[temp5][0]
        except:
            labIndex = input_data[m][0][0][0]
            # Gets the eigenvalue corresponding to the subscript index
            indexNum = list(map_data).index(labIndex)
            valueAll[0, indexNum] = None
            continue
    pass

# med
def get_med(input_data, input_t_time):
    for m in range(len(input_data)):
        try:
            labIndex = input_data[m][0][0][0]
            # Gets the eigenvalue corresponding to the subscript index
            indexNum = list(map_data).index(labIndex)
            temp = input_data[m][1]
            temp1 = np.asarray(temp)
            temp2 = temp1[:, -1]
            temp2 = list(map(float, temp2))
            temp3 = [x for x in temp2 if x <= input_t_time]
            temp4 = np.max(temp3)
            temp2.reverse()
            temp5 = temp2.index(temp4)
            temp5 = len(temp2) - 1 - temp5
            valueAll[0, indexNum] = temp[temp5][0]
        except:
            continue
    pass

# ccs
def get_ccs_px(input_data, input_t_time):
    for m in range(len(input_data)):
        try:
            ccsTimes = input_data[m][1]
            ccsTimes = list(map(float, ccsTimes))
            ccsTime = np.min(ccsTimes)
            if ccsTime <= input_t_time:
                # Gets the eigenvalue corresponding to the subscript index
                ccsIndex = input_data[m][0][0]
                indexNum = list(map_data).index(ccsIndex)
                valueAll[0][indexNum] = 1
        except:
            continue
    pass

# label
def get_label(labels, advance_day):
    # Gets the eigenvalue corresponding to the subscript index
    day_index = list(map_data).index("days")
    value_index = list(map_data).index("AKI_label")
    for AKI_data in labels:
        if float(AKI_data[0]) > 0:
            valueAll[0, day_index] = float(AKI_data[1]) - advance_day
            valueAll[0, value_index] = 1
            break
        else:
            valueAll[0, day_index] = float(AKI_data[1]) - advance_day
            valueAll[0, value_index] = 0
    return valueAll[0, value_index], valueAll[0, day_index]

"""
读取list数据，转换为array数据（samples）
输入参数：年份（要求相应数据存在）
"""
print('Begin...')
year = sys.argv[1]
string_list_file_path = "/home/xzhang_sta/work/yuanborong/data/row_data/list-" + str(year) + ".pkl" # list数据路径
map_file_path = "/home/xzhang_sta/work/yuanborong/data/row_data/feature_dict_BDAI_map.pkl"         # feature_dict_BDAI_map.pkl文件的路径
save_file_path = "/panfs/pfs.local/work/liu/xzhang_sta/yuanborong/data_new/df_data/pre_48h/data-" + str(year) + ".pkl"        # 生成数据保存的路径
pre_day = 2  # How much time in advance

f = open(string_list_file_path, 'rb')
data = pickle.load(f)
map_f = open(map_file_path, 'rb')
map_data = pickle.load(map_f)

Data = []
throw_num = 0
for i in range(len(data)):
    valueAll = np.zeros([1, len(map_data)])  # all feature number is 28299
    demo, vital, lab, ccs, px, med, label = data[i]
    AKI_status, pre_time = get_label(label, pre_day)
    if pre_time < 0:
        throw_num += 1
        print("Discard samples AKI label", [AKI_status, pre_time])
        continue
    get_demo(demo)
    get_vital(vital, pre_time)
    get_lab(lab, pre_time)
    get_med(med, pre_time)
    get_ccs_px(ccs, pre_time)
    get_ccs_px(px, pre_time)
    if len(Data) == 0:
        Data = valueAll
    else:
        Data = np.row_stack((Data, valueAll))

# save data
print("throw sample number:", throw_num)
print("Data size:", Data.shape)
print("positive samples size:", np.sum(Data[:, -1]))
print("negative samples size:", len(Data) - np.sum(Data[:, -1]))
print("The incidence of AKI:", np.average(Data[:, -1]))
Data = pd.DataFrame(Data)
Data.columns = map_data
Data.to_pickle(save_file_path)
print("data saved in {}".format(save_file_path))
# -*- coding: utf-8 -*-#
"""
This is the common function used in the program

"""
import os
import pickle
import numpy as np
import pandas as pd

data_list = [0] * 50000
fixed_missing_rate = 0.9999
dtrain = []


# Returns a list of files, including the filename (full path)
def get_filelist(dir):
    filelist = []
    if os.path.isfile(dir):
        filelist.append(dir)
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newDir = os.path.join(dir, s)
            filelist.extend(get_filelist(newDir))
    return filelist


def get_map(path):
    map_f = open(path, 'rb')
    map_data = pickle.load(map_f)
    map_f.close()
    return map_data


# demo
def get_demo(dat, map_data, value_all):
    temp = np.array(dat)
    # Gets the eigenvalue corresponding to the subscript index
    for m in range(len(temp)):
        demo_index = "demo"
        if m == 0:
            demo_index = demo_index + str(1)
        else:
            demo_index = demo_index + str(m + 1) + str(temp[m])
            temp[m] = 1

        # index_num = list(map_data).index(demo_index)
        index_num = map_data[demo_index]
        value_all[0, index_num] = temp[m]
    return value_all


# vital
def get_vital(dat, t, map_data, value_all):
    for m in range(len(dat)):
        try:
            temp = np.asarray(dat[m]).astype(float)
            meas_t = np.array(temp)[:, -1].astype(float)
            threshold = np.min(meas_t) - 1
            meas_t[meas_t > t] = threshold  # filter data, which is after occur_time
            if np.max(meas_t) == threshold:
                continue
            nearest_t = np.where(meas_t == meas_t.max())[0][-1]
            if m == 3 or m == 4 or m == 5:
                # vital4,vital5,vital6 is a qualitative variable
                # Gets the eigenvalue corresponding to the subscript index
                vital_index = 'vital' + str((m + 1) * 10) + str(int(temp[nearest_t][0]))
                # index_num = list(map_data).index(vital_index)
                index_num = map_data[vital_index]
                value_all[0, index_num] = 1
                continue
            else:
                if m == 6 or m == 7:
                    tw = 1  # Time window 24 hours from the measured value
                    meas_t[meas_t < meas_t[
                        nearest_t] - tw] = threshold  # Set the data out of the window less than time to 0
                    meas_t_max = max(meas_t)
                    if meas_t_max == threshold:
                        continue
                    meas_t_min = min(filter(lambda x: x > threshold, meas_t))

                    meas_index = np.where(meas_t > threshold)[0]  # Find an index with a non-zero value
                    meas_bp = np.array(temp)[:, 0].astype(float)
                    meas_bp_max, meas_bp_min = max(meas_bp[meas_index]), min(meas_bp[meas_index])

                    bp_min = meas_bp_min
                    if meas_t_max - meas_t_min != 0:
                        bp_slp = round((meas_bp_max - meas_bp_min) / (meas_t_max - meas_t_min),
                                       2)  # If there is only one value, bp_slp is NAN
                    else:
                        bp_slp = 0

                    if m == 6:  # BP_SYSTOLIC
                        temp[nearest_t][0] = temp[nearest_t][0] \
                            if 40 <= temp[nearest_t][0] <= 210 else 0
                    if m == 7:  # BP_DIASTOLIC
                        temp[nearest_t][0] = temp[nearest_t][0] \
                            if temp[nearest_t][0] <= 120 or temp[nearest_t][0] >= 40 else 0

                    value_all[0, map_data['bp_min']] = bp_min
                    value_all[0, map_data['bp_slp']] = bp_slp

                # Gets the eigenvalue corresponding to the subscript index
                vital_index = 'vital' + str(m + 1)
                # index_num = list(map_data).index(vital_index)
                index_num = map_data[vital_index]
                if m == 0:  # HT
                    temp[nearest_t][0] = temp[nearest_t][0] \
                        if 0 < temp[nearest_t][0] <= 95 else 0
                if m == 1:  # WT
                    temp[nearest_t][0] = temp[nearest_t][0] \
                        if 0 < temp[nearest_t][0] <= 1400 else 0
                if m == 2:  # BMI
                    temp[nearest_t][0] = temp[nearest_t][0] \
                        if 0 < temp[nearest_t][0] <= 70 else 0

                value_all[0, index_num] = temp[nearest_t][0]
        except Exception as e:
            # print("vital handle error:", e)
            continue
    return value_all


# lab
def get_lab(dat, t, map_data, value_all):
    keys = map_data.keys()
    for m in range(len(dat)):
        try:
            labIndex = dat[m][0][0][0]
            # Gets the eigenvalue corresponding to the subscript index
            # index_num = list(map_data).index(labIndex)
            index_num = map_data[labIndex]
            temp = dat[m][1]
            meas_t = np.array(temp)[:, -1].astype(float)
            threshold = np.min(meas_t) - 1
            meas_t[meas_t > t] = threshold  # filter data, which is after occur_time
            if np.max(meas_t) == threshold:
                continue
            nearest_t = np.where(meas_t == meas_t.max())[0][-1]
            if labIndex + "|change" in keys:
                if nearest_t - 1 >= 0:
                    scr_change = float(temp[nearest_t][0]) - float(temp[nearest_t - 1][0])
                else:
                    scr_change = np.nan
                value_all[0, map_data[labIndex + "|change"]] = scr_change

            value_all[0, index_num] = temp[nearest_t][0]
        except Exception as e:
            # print("lab handle error:", e)
            continue
    return value_all


# med
def get_med(dat, t, map_data, value_all):
    for m in range(len(dat)):
        try:
            medIndex = dat[m][0][0][0]
            # Gets the eigenvalue corresponding to the subscript index
            # index_num = list(map_data).index(labIndex)
            index_num = map_data[medIndex]
            temp1 = np.asarray(dat[m][1]).astype(float)
            temp2 = temp1[:, -1]
            temp3 = [x for x in temp2 if x <= t]
            if len(temp3) == 0:
                continue
            value_all[0, index_num] = len(temp3)
        except Exception as e:
            # print("med handle error:", e)
            # print("med missing:", dat[m][0][0][0])
            continue
    return value_all


# ccs
def get_ccs(dat, t, map_data, value_all):
    for m in range(len(dat)):
        try:
            ccsTimes = dat[m][1]
            ccsTimes = list(map(float, ccsTimes))
            ccsTime = np.min(ccsTimes)
            if ccsTime <= t:
                # Gets the eigenvalue corresponding to the subscript index
                ccsIndex = dat[m][0][0]
                # index_num = list(map_data).index(ccsIndex)
                index_num = map_data[ccsIndex]
                value_all[0][index_num] = 1
        except Exception as e:
            # print("ccs handle error:", e)
            # print("ccs missing:", dat[m][0][0])
            continue
    return value_all


# px
def get_px(dat, t, map_data, value_all):
    for m in range(len(dat)):
        try:
            pxTimes = dat[m][1]
            pxTimes = list(map(float, pxTimes))
            pxTime = np.min(pxTimes)
            if pxTime <= t:
                # Gets the eigenvalue corresponding to the subscript index
                pxIndex = dat[m][0][0]
                # index_num = list(map_data).index(ccsIndex)
                index_num = map_data[pxIndex]
                value_all[0][index_num] = 1
        except Exception as e:
            # print("px handle error:", e)
            # print("px missing:", dat[m][0][0])
            continue
    return value_all


def get_instance(list_num):
    demo_vital_num, lab_num, ccs_px_num, med_num, new_feature_num = list_num
    demo_vital = np.zeros([1, demo_vital_num])
    lab = np.full([1, lab_num], np.nan)
    ccs_px = np.zeros([1, ccs_px_num])
    med = np.full([1, med_num], np.nan)
    new_feature = np.zeros([1, new_feature_num], dtype=np.float32)
    return np.hstack((demo_vital, lab, ccs_px, med, new_feature))


def get_DataFrame(data, list_num):
    demo_vital_num, lab_num, ccs_px_num, med_num, new_feature_num = list_num
    d_v_start, d_v_end = 0, demo_vital_num
    lab_start, lab_end = demo_vital_num, demo_vital_num + lab_num
    c_p_start, c_p_end = demo_vital_num + lab_num, demo_vital_num + lab_num + ccs_px_num
    med_start, med_end = demo_vital_num + lab_num + ccs_px_num, demo_vital_num + lab_num + ccs_px_num + med_num
    n_f_start, n_f_end = demo_vital_num + lab_num + ccs_px_num + med_num, demo_vital_num + lab_num + ccs_px_num + med_num + new_feature_num
    demo_vital_df = pd.DataFrame(data[:, d_v_start:d_v_end], dtype=np.float16)
    lab_df = pd.DataFrame(data[:, lab_start:lab_end], dtype=np.float32)
    ccs_px_df = pd.DataFrame(data[:, c_p_start:c_p_end], dtype=np.int8)
    med_df = pd.DataFrame(data[:, med_start:med_end], dtype=np.int8)
    new_feature_df = pd.DataFrame(data[:, n_f_start:n_f_end], dtype=np.float32)
    dataframe = pd.concat([demo_vital_df, lab_df, ccs_px_df, med_df, new_feature_df], axis=1)
    return dataframe


def saveData(column, data, save_path, list_num):
    sample_num = 0
    for i in range(len(data)):
        sample_num += len(data[i][-1])
    result_data = np.zeros([sample_num, column], dtype=np.float32)
    index = 0
    for i in range(len(data)):
        for j in range(len(data[i][-1])):
            result_data[index] = data[i][-1][j]
            index += 1
    result_data = get_DataFrame(result_data, list_num)
    print("data loaded.", result_data.shape)
    print("labels:")
    print(result_data.iloc[:, -2].value_counts())
    isExists = os.path.exists(save_path)
    if not isExists:
        os.makedirs(save_path)
    result_data.to_csv(save_path + '/data.csv', index=False)


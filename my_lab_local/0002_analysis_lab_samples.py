# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     get_some_samples
   Description:   ...
   Author:        cqh
   date:          2022/5/11 9:52
-------------------------------------------------
   Change Activity:
                  2022/5/11:
-------------------------------------------------
"""
__author__ = 'cqh'

import time

import joblib
import pandas as pd
from my_logger import MyLog
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
from threading import Lock
import os


def get_lab_and_unit(e_id, lab_input_data):
    """
    lab_input_data
    [
      [
        [
          ['lab65']
        ],
        [
          ['0','K/UL','16'],
          [ ... ]
        ]
      ],
      ...
    ]
    :param e_id: 病人ID
    :param lab_input_data:
    :return:
    """
    lab_detail_list = []
    for m in range(len(lab_input_data)):
        try:
            # lab_id
            lab_id = lab_input_data[m][0][0][0]
            # lab_details
            lab_details = lab_input_data[m][1]

            for detail in lab_details:
                lab_one_detail = [lab_id, detail[0], detail[1], detail[2]]
                lab_detail_list.append(lab_one_detail)
        except Exception as err:
            my_logger.error(err)
            continue

    if lab_detail_list is None:
        return

    my_logger.info(f"encounter_id:{e_id} | len:{len(lab_detail_list)}")
    lab_detail_list_df = pd.DataFrame(lab_detail_list, columns=columns)
    global lab_all_sample_df
    global write_lock
    try:
        write_lock.acquire()
        lab_all_sample_df = pd.concat([lab_all_sample_df, lab_detail_list_df], ignore_index=True)
    except Exception as err:
        my_logger.error(err)
    finally:
        write_lock.release()


def multi_thread_process():
    my_logger.warning("start process ... ")
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=pool_nums) as executor:
        thread_list = []
        for i in range(len(list_data)):
            encounter_id = list_data[i][0]
            lab_list_data = list_data[i][3]
            thread = executor.submit(get_lab_and_unit, encounter_id, lab_list_data)
            thread_list.append(thread)
        wait(thread_list, return_when=ALL_COMPLETED)

    run_time = round(time.time() - start_time, 2)
    my_logger.warning(
        f"process lab feature need: {run_time} s")


def analysis_lab_unit():
    lab_all_sample = pd.read_csv(lab_csv_save_path)
    if lab_all_sample is None:
        my_logger.error("Can't find lab  csv file to read!")

    my_logger.info(f"shape: {lab_all_sample.shape}")

    # 找到不重复的lab_id列表
    lab_id_unique = lab_all_sample['lab_id'].unique()
    lab_id_unit_count_list = []
    for lab_id in lab_id_unique:
        # lab_id对应的数据
        lab_data_id = lab_all_sample.loc[lab_all_sample['lab_id'] == lab_id]
        lab_id_unit_count = [lab_id, lab_data_id['unit'].unique().size]
        lab_id_unit_count_list.append(lab_id_unit_count)

    count_lab_id_unit_df = pd.DataFrame(lab_id_unit_count_list, columns=('lab_id', 'unit_count'))
    # 选出前10多单位数量的lab特征ID
    lab_id_top_10 = count_lab_id_unit_df.sort_values(by=['unit_count'], ascending=False).head(10)['lab_id']
    # 得到前10个lab的单位分布
    lab_data_unit_result = {}
    for lab_id in lab_id_top_10:
        # lab_id对应的数据
        lab_data_id = lab_all_sample.loc[lab_all_sample['lab_id'] == lab_id]
        # lab_id对应单位分布
        lab_data_unit_count = lab_data_id['unit'].value_counts().to_frame().astype('float64').sort_values(by=['unit'],
                                                                                                          ascending=False)
        lab_data_unit_count['percent'] = lab_data_unit_count.div(lab_data_unit_count.sum())
        lab_data_unit_result[lab_id] = lab_data_unit_count

    my_logger.warning(f"lab_data_unit_result keys:{lab_data_unit_result.keys()}")
    # save result to pkl (dict format)
    joblib.dump(lab_data_unit_result, lab_result_pkl_save_path)
    my_logger.warning(f"save to pkl success!")


if __name__ == '__main__':
    BASE_PATH = "D:\\lab\\other_file\\"
    string_list_file_path = "D:\\lab\\other_file\\2016_string2list.pkl"
    lab_csv_save_path = os.path.join(BASE_PATH, "lab_all_sample.csv")
    lab_result_pkl_save_path = os.path.join(BASE_PATH, "lab_data_unit_result.pkl")
    columns = ['lab_id', 'value', 'unit', 'day']

    pool_nums = 10

    my_logger = MyLog().logger
    list_data = joblib.load(string_list_file_path)
    # 公有变量
    lab_all_sample_df = pd.DataFrame(columns=columns)
    write_lock = Lock()
    my_logger.warning("starting to process lab data and save to csv...")
    multi_thread_process()
    lab_all_sample_df.to_csv(lab_csv_save_path, index=False)
    my_logger.warning("starting to save the result to pkl...")
    analysis_lab_unit()

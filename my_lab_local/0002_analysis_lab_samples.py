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


def get_lab_and_unit(idx, lab_input_data):
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
    :param idx: 第idx个病人
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

    my_logger.info(f"idx:{idx} | len:{len(lab_detail_list)}")
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
            lab_list_data = list_data[i][3]
            thread = executor.submit(get_lab_and_unit, i, lab_list_data)
            thread_list.append(thread)
        wait(thread_list, return_when=ALL_COMPLETED)

    run_time = round(time.time() - start_time, 2)
    my_logger.warning(
        f"process lab feature need: {run_time} s")


if __name__ == '__main__':
    string_list_file_path = "D:\\lab\\other_file\\2016_string2list.pkl"
    save_path = "D:\\lab\\other_file\\lab_all_sample.csv"
    columns = ['lab_id', 'value', 'unit', 'day']
    pool_nums = 10
    my_logger = MyLog().logger
    list_data = joblib.load(string_list_file_path)
    # 公有变量
    lab_all_sample_df = pd.DataFrame(columns=columns)
    write_lock = Lock()
    multi_thread_process()
    lab_all_sample_df.to_csv(save_path, index=False)

# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     multi_process_share_memory
   Description:   ...
   Author:        cqh
   date:          2022/5/10 19:44
-------------------------------------------------
   Change Activity:
                  2022/5/10:
-------------------------------------------------
"""
__author__ = 'cqh'

import time

from my_logger import MyLog
from multiprocessing import shared_memory, Pool, Manager
import numpy as np
import pandas as pd

len_list_data = 2000
len_s_map_data = 1000
my_logger = MyLog().logger


def process_cur_sample(idx, share_memory):
    shared_all_sample = shared_memory.SharedMemory(name='all_sample')
    all_sample = np.ndarray(shape=share_memory, dtype=np.float64, buffer=shared_all_sample.buf)
    all_sample[idx, :] = idx
    time.sleep(0.1)
    shared_all_sample.close()


def process_cur_sample2(idx, namespace, write_lock):
    write_lock.acquire()
    all_sample2 = namespace.all_sample2
    all_sample2[idx, :] = idx
    namespace.all_sample2 = all_sample2
    write_lock.release()

    time.sleep(0.1)


if __name__ == '__main__':

    all_sample = np.zeros((len_list_data, len_s_map_data))
    sm_all_sample = shared_memory.SharedMemory(name='all_sample', create=True, size=all_sample.nbytes)
    s_all_sample = np.ndarray(shape=all_sample.shape, dtype=np.float64, buffer=sm_all_sample.buf)
    s_all_sample[:, :] = all_sample[:, :]
    shape_all_sample = all_sample.shape

    pool = Pool(processes=6)
    start_time = time.time()
    # 多线程1
    for i in range(len_list_data):
        pool.apply_async(func=process_cur_sample,
                         args=(i, shape_all_sample))

    pool.close()
    pool.join()

    my_logger.info(f"mult time:{time.time() - start_time} s")
    result = pd.DataFrame(s_all_sample)
    my_logger.info(result.describe())

    # 多线程2
    pool2 = Pool(processes=6)
    global_manager = Manager()
    global_namespace = global_manager.Namespace()
    lock = global_manager.Lock()
    all_sample2 = np.zeros((len_list_data, len_s_map_data))
    global_namespace.all_sample2 = all_sample2
    start_time2 = time.time()
    for i in range(len_list_data):
        pool2.apply_async(func=process_cur_sample2,
                          args=(i, global_namespace, lock))
    pool2.close()
    pool2.join()
    my_logger.info(f"mult2 time:{time.time() - start_time2} s")
    result2 = pd.DataFrame(global_namespace.all_sample2)
    my_logger.info(result2.describe())

    # 单线程
    all_sample3 = np.zeros((len_list_data, len_s_map_data))
    start_time3 = time.time()
    for i in range(len_list_data):
        all_sample3[i, :] = i
        time.sleep(0.1)
    my_logger.info(f"for time:{time.time() - start_time3} s")
    result3 = pd.DataFrame(all_sample3)
    my_logger.info(result3.describe())

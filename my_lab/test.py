# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     test
   Description:   ...
   Author:        cqh
   date:          2022/4/22 18:00
-------------------------------------------------
   Change Activity:
                  2022/4/22:
-------------------------------------------------
"""
__author__ = 'cqh'

from multiprocessing import Pool, Manager
import pandas as pd
import time
from my_logger import MyLog
import os
import psutil

n_personal_model_each_iteration = 1000
my_logger = MyLog().logger
columns = ['aaa', 'bbb', 'ccc']


def run(x, global_ns, lock):
    start_time = time.time()

    try:
        lock.acquire()
        iter_data = global_ns.iteration_data
        iter_data.iloc[x, :] = x
        global_ns.iteration_data = iter_data

        iter_y = global_ns.iteration_y
        iter_y[x] = x
        global_ns.iteration_y = iter_y

    except Exception as err:
        raise err
    finally:
        lock.release()
    memory_used = round(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024, 2)
    my_logger.info(f"pid:{os.getpid()} | memory used:{memory_used} MB")


if __name__ == '__main__':

    for i in range(100):
        manager = Manager()
        lock = manager.Lock()
        global_namespace = manager.Namespace()
        global_namespace.iteration_data = pd.DataFrame(index=range(n_personal_model_each_iteration), columns=columns)
        global_namespace.iteration_y = pd.Series(index=range(n_personal_model_each_iteration), dtype='float64')
        # del iteration_data
        # del iteration_y

        pool = Pool(processes=6)
        st_t = time.time()
        for j in range(n_personal_model_each_iteration):
            pool.apply_async(func=run, args=(j, global_namespace, lock))

        pool.close()
        pool.join()

        my_logger.info(f"all_time: {time.time() - st_t}")
        print(global_namespace.iteration_data)
        print(global_namespace.iteration_y)

        del global_namespace

        memory_used = round(psutil.Process(os.getpid()).memory_info().rss / 1024 /  1024, 2)
        my_logger.info(f"[for]:pid:{os.getpid()} | memory used:{memory_used} MB")



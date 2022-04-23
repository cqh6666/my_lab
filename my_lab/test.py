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

from multiprocessing import Pool, Manager, Lock, Process, Array
import pandas as pd
import time
from my_logger import MyLog
import os
import tracemalloc

n_personal_model_each_iteration = 1000
my_logger = MyLog().logger
columns = ['aaa', 'bbb', 'ccc']


def run(x, global_ns, lock):
    start_time = time.time()
    my_logger.info(global_ns.list)

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
    my_logger.info(f"time: {time.time() - start_time}")


if __name__ == '__main__':

    tracemalloc.start()

    # iteration_data =
    # iteration_y =
    manager = Manager()
    lock = manager.Lock()
    global_namespace = manager.Namespace()
    global_namespace.iteration_data = pd.DataFrame(index=range(n_personal_model_each_iteration), columns=columns)
    global_namespace.iteration_y = pd.Series(index=range(n_personal_model_each_iteration), dtype='float64')
    global_namespace.list = range(0,1000)
    # del iteration_data
    # del iteration_y

    pool = Pool(processes=6)
    st_t = time.time()
    for i in range(n_personal_model_each_iteration):
        pool.apply_async(func=run, args=(i, global_namespace, lock))

    pool.close()
    pool.join()

    my_logger.info(f"all_time: {time.time() - st_t}")
    print(global_namespace.iteration_data)
    print(global_namespace.iteration_y)

    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage is {current / 10 ** 6}MB; Peak was {peak / 10 ** 6}MB")
    tracemalloc.stop()

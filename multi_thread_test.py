# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     multi_thread_test
   Description:   ...
   Author:        cqh
   date:          2022/5/17 14:44
-------------------------------------------------
   Change Activity:
                  2022/5/17:
-------------------------------------------------
"""
__author__ = 'cqh'

from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED, as_completed
import threading
import time
import numpy as np

def do_something():
    print(f"do_something_current_thread:{threading.current_thread().getName()}")


def run(idd):
    do_something()
    print(f"run_current_thread:{threading.current_thread().getName()}")
    return idd


if __name__ == '__main__':

    print(f"main_current_thread:{threading.current_thread().getName()}")

    pg_start_time = time.time()
    pool_nums = 6

    result_list = []
    with ThreadPoolExecutor(max_workers=pool_nums) as executor:
        thread_list = []
        for idx in range(10000):
            thread = executor.submit(run, idx)
            thread_list.append(thread)

        for thread in as_completed(thread_list):
            result_list.append(thread.result())

    result = np.mean(result_list)
        # wait(thread_list, return_when=ALL_COMPLETED)

    print(f"main_run_time:{time.time() - pg_start_time} s")

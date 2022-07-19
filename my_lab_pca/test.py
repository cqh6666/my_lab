# -*- encoding:gbk -*-
"""
-------------------------------------------------
   File Name:     test
   Description:   ...
   Author:        cqh
   date:          2022/7/15 10:43
-------------------------------------------------
   Change Activity:
                  2022/7/15:
-------------------------------------------------
"""
__author__ = 'cqh'

# Importing memory-profiler module in the program
import csv
import time
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
import pandas as pd
from memory_profiler import profile
import objgraph
import numpy as np



def f2(n):
    c = n * 10
    return c

# Profile Decorator class
@profile
# A default function to check memory usage
def defFunc():
    a = [1] * (10 ** 6)
    b = [2] * (2 * 10 ** 7)
    c = f2(b)
    objgraph.show_refs([a,b,c], too_many=5, filename='too-many.png')
    del c

    return a

@profile
def defFunc2():
    x = []
    y = [x, [x], dict(x=x)]
    roots = objgraph.get_leaking_objects()
    objgraph.show_most_common_types(objects=roots)
    return y

if __name__ == '__main__':
    # Calling default function
    defFunc()
    pool_nums = 10
    # 匹配相似样本（从训练集） XGB建模 多线程
    with ThreadPoolExecutor(max_workers=pool_nums) as executor:
        thread_list = []
        for i in range(5):
            defFunc()
        wait(thread_list, return_when=ALL_COMPLETED)
    # Print confirmation message
    print("We have successfully inspected memory usage from the default function!")
    a = defFunc()




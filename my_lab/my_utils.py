# -*- coding: gbk -*-
"""
-------------------------------------------------
   File Name:     utils_function
   Description:   ...
   Author:        cqh
   date:          2022/4/22 17:20
-------------------------------------------------
   Change Activity:
                  2022/4/22:
-------------------------------------------------
"""
__author__ = 'cqh'

from my_logger import MyLog
from time import perf_counter


def coast_time(func):
    def fun(*args, **kwargs):
        t = perf_counter()
        result = func(*args, **kwargs)
        MyLog().logger.info(f'func {func.__name__} coast time: {perf_counter() - t:.8f} s')
        return result

    return fun


@coast_time
def run():
    print("test")


if __name__ == '__main__':

    run()
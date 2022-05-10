# encoding=gbk
import multiprocessing as mp

import time
import numpy as np
import pandas as pd
import psutil
from my_logger import MyLog
import os

my_logger = MyLog().logger


def my_worker():
    use = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024
    use_per = psutil.Process(os.getpid()).memory_percent()
    my_logger.info(use)
    my_logger.info(use_per)


def in_test():
    a = ["111_", "222@", "333-"]
    str_a = "0009_1_111"
    flag = "111_"
    if flag in str_a:
        return True

    return False


if __name__ == '__main__':
    print(in_test())

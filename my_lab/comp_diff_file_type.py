# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     comp_diff_file_type
   Description:   ...
   Author:        cqh
   date:          2022/6/10 21:34
-------------------------------------------------
   Change Activity:
                  2022/6/10:
-------------------------------------------------
"""
__author__ = 'cqh'
import os
import time
import pandas as pd
from my_logger import MyLog

my_logger = MyLog().logger

def compare_file_type(feather, csv, pickle):
    """
    比较不同保存格式的读写时间和文件大小
    :param feather:
    :param csv:
    :param pickle:
    :return:
    """
    feather_size = hum_convert(os.path.getsize(feather))
    csv_size = hum_convert(os.path.getsize(csv))
    pickle_size = hum_convert(os.path.getsize(pickle))
    my_logger.info(f"feather size: {feather_size}")
    my_logger.info(f"csv size: {csv_size}")
    my_logger.info(f"pickle size: {pickle_size}")

    start_time = time.time()
    pd.read_feather(feather)
    feather_time = time.time()
    pd.read_csv(csv)
    csv_time = time.time()
    pd.read_pickle(pickle)
    pickle_time = time.time()
    my_logger.info(f"load feather time: {feather_time - start_time}")
    my_logger.info(f"load csv time: {csv_time - feather_time}")
    my_logger.info(f"load pickle time: {pickle_time - csv_time}")


def hum_convert(value):
    """
    自动换算为比较恰当的文件大小单位
    :param value:
    :return:
    """
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    size = 1024.0
    for i in range(len(units)):
        if (value / size) < 1:
            return "%.2f%s" % (value, units[i])
        value = value / size
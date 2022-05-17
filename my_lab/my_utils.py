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
import os


def humansize(size):
    """将文件的大小转成带单位的形式
    >>> humansize(1024) == '1 KB'
    True
    >>> humansize(1000) == '1000 B'
    True
    >>> humansize(1024*1024) == '1 M'
    True
    >>> humansize(1024*1024*1024*2) == '2 G'
    True
    """
    units = ['B', 'KB', 'M', 'G', 'T']
    for unit in units:
        if size < 1024:
            break
        size = size // 1024
    return '{} {}'.format(size, unit)


def humantime(seconds):
    """将秒数转成00:00:00的形式
    >>> humantime(3600) == '01:00:00'
    True
    >>> humantime(360) == '06:00'
    True
    >>> humantime(6) == '00:06'
    True
    """
    h = m = ''
    seconds = int(seconds)
    if seconds >= 3600:
        h = '{:02}:'.format(seconds // 3600)
        seconds = seconds % 3600
    if 1 or seconds >= 60:
        m = '{:02}:'.format(seconds // 60)
        seconds = seconds % 60
    return '{}{}{:02}'.format(h, m, seconds)


def get_cpu_info():
    pid = os.getpid()
    from psutil import cpu_count, cpu_percent, virtual_memory, Process
    python_process = Process(pid)
    cpu_info_dict = {
        "cpu_count": cpu_count(),
        "cpu_percent": cpu_percent(),
        "memory_total": humansize(virtual_memory().total),
        "memory_used": virtual_memory().percent
    }
    return cpu_info_dict


if __name__ == '__main__':
    dic = get_cpu_info()
    print([i for i in range(1,100,2)])
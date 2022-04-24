# -*- coding: gbk -*-
"""
-------------------------------------------------
   File Name:     my_logger
   Description:   ...
   Author:        cqh
   date:          2022/4/21 15:40
-------------------------------------------------
   Change Activity:
                  2022/4/21:
-------------------------------------------------
"""
__author__ = 'cqh'

import sys
import logging
from time import strftime

# 保存路径
# log_file_name = f'/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/result/cqh_{strftime("%Y-%m-%d")}.log'
log_file_name = f'./log/cqh_{strftime("%Y-%m-%d")}.log'
# 设置日志格式#和时间格式
FMT = '%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s: %(message)s'
DATEFMT = '%Y-%m-%d %H:%M:%S'


class MyLog(object):
    def __init__(self):
        self.logger = logging.getLogger()
        self.formatter = logging.Formatter(fmt=FMT, datefmt=DATEFMT)
        self.log_filename = log_file_name

        self.logger.addHandler(self.get_file_handler(self.log_filename))
        self.logger.addHandler(self.get_console_handler())
        # 设置日志的默认级别
        self.logger.setLevel(logging.ERROR)

    # 输出到文件handler的函数定义
    def get_file_handler(self, filename):
        filehandler = logging.FileHandler(filename, encoding="gbk")
        filehandler.setFormatter(self.formatter)
        return filehandler

    # 输出到控制台handler的函数定义
    def get_console_handler(self):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(self.formatter)
        return console_handler




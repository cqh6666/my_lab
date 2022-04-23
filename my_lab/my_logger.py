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

import os
import sys
import logging
from time import strftime

# ����·��
PATH = os.path.abspath('.') + '/log/'
# ������־��ʽ#��ʱ���ʽ
FMT = '%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s: %(message)s'
DATEFMT = '%Y-%m-%d %H:%M:%S'


class MyLog(object):
    def __init__(self):
        self.logger = logging.getLogger()
        self.formatter = logging.Formatter(fmt=FMT, datefmt=DATEFMT)
        self.log_filename = '{0}{1}.log'.format(PATH, strftime("%Y-%m-%d"))

        self.logger.addHandler(self.get_file_handler(self.log_filename))
        self.logger.addHandler(self.get_console_handler())
        # ������־��Ĭ�ϼ���
        self.logger.setLevel(logging.DEBUG)

    # ������ļ�handler�ĺ�������
    def get_file_handler(self, filename):
        filehandler = logging.FileHandler(filename, encoding="utf-8")
        filehandler.setFormatter(self.formatter)
        return filehandler

    # ���������̨handler�ĺ�������
    def get_console_handler(self):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(self.formatter)
        return console_handler




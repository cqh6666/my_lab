# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     test_log
   Description:   ...
   Author:        cqh
   date:          2022/5/10 16:50
-------------------------------------------------
   Change Activity:
                  2022/5/10:
-------------------------------------------------
"""
__author__ = 'cqh'

from my_logger import MyLog


def run():
    my_logger.info("sd")


def run_error():
    raise Exception('error!')


if __name__ == '__main__':
    my_logger = MyLog().logger
    run()
    run_error()

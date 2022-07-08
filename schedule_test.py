# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     schedule_test
   Description:   ...
   Author:        cqh
   date:          2022/6/20 20:10
-------------------------------------------------
   Change Activity:
                  2022/6/20:
-------------------------------------------------
"""
__author__ = 'cqh'
import schedule
import time
import os


def run():
    print("hello!")

schedule.every(1).minutes.do(run)

while True:
    schedule.run_pending()
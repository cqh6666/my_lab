# -*- coding: gbk -*-
"""
-------------------------------------------------
   File Name:     0009_schedule_submit
   Description:   ...
   Author:        cqh
   date:          2022/5/6 16:12
-------------------------------------------------
   Change Activity:
                  2022/5/6:
-------------------------------------------------
"""
__author__ = 'cqh'

import schedule
import time


def submit_job(n):
    print("working...", n)


if __name__ == '__main__':
    # schedule.every(1).minutes.do(submit_job,1)
    #
    # print("start schedule...")
    # while True:
    #     schedule.run_pending()
    #     time.sleep(1)
    pass
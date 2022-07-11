# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     MyTimer
   Description:   ...
   Author:        cqh
   date:          2022/7/8 15:27
-------------------------------------------------
   Change Activity:
                  2022/7/8:
-------------------------------------------------
"""
__author__ = 'cqh'

import time
from utils_api import covert_time_format


class MyTimer:
    def __init__(self, func=time.perf_counter):
        self.elapsed = 0.0
        self._func = func
        self._start = None

    def start(self):
        if self._start is not None:
            raise RuntimeError('Already started')
        self._start = self._func()

    def stop(self):
        if self._start is None:
            raise RuntimeError('Not started')
        end = self._func()
        self.elapsed += end - self._start
        self._start = None

    def reset(self):
        self.elapsed = 0.0

    def show_time(self):
        return "cost time: {}".format(covert_time_format(self.elapsed))

    @property
    def running(self):
        return self._start is not None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


if __name__ == '__main__':
    my_time = MyTimer()

    my_time.start()
    time.sleep(2)
    my_time.stop()

    print(my_time.show_time())

    with MyTimer() as mt:
        time.sleep(2)


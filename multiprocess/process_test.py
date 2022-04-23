# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     process_test
   Description :
   Author :       cqh
   date：          2022/4/13 11:00
-------------------------------------------------
   Change Activity:
                   2022/4/13:
-------------------------------------------------
"""
__author__ = 'cqh'

from multiprocessing import Process


def func1(name):
    print('多进程%s ' %name )

if __name__ == '__main__':
    process_list = []
    for i in range(5):  #开启5个子进程执行fun1函数
        p = Process(target=func1,args=('Python',)) #实例化进程对象
        p.start()
        process_list.append(p)

    for i in process_list:
        p.join()

    print('结束测试')
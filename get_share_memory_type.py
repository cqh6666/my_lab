# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     get_share_memory_type
   Description:   ...
   Author:        cqh
   date:          2022/5/10 21:09
-------------------------------------------------
   Change Activity:
                  2022/5/10:
-------------------------------------------------
"""
__author__ = 'cqh'
import numpy as np
from multiprocessing import shared_memory

len_list_data = 2000
len_s_map_data = 1000
# init a variable to save all eligible cross-section sample 初始化数据集格式
all_sample = np.zeros((len_list_data, len_s_map_data))

# create shared_memory for all_sample 创建共享内存变量 all_sample
sm_all_sample = shared_memory.SharedMemory(name='all_sample', create=True, size=all_sample.nbytes)
# 生成
buf = sm_all_sample.buf
buf[1]=bytes(55)
print(buf)
s_all_sample = np.ndarray(shape=all_sample.shape, dtype=np.float64, buffer=sm_all_sample.buf)
s_all_sample[:, :] = all_sample[:, :]
s_all_sample[0, 0] = 99
shape_all_sample = all_sample.shape


sb = shared_memory.SharedMemory(sm_all_sample.name)
s_all_sample_b = np.ndarray(shape=all_sample.shape, dtype=np.float64, buffer=sb.buf)


# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     pg_data
   Description:   ...
   Author:        cqh
   date:          2022/4/22 21:52
-------------------------------------------------
   Change Activity:
                  2022/4/22:
-------------------------------------------------
"""
__author__ = 'cqh'
import pandas as pd
import os

# (122471, 4129)
train_data_x_file = os.path.join('../feather/x_train.feather')
train_x = pd.read_feather(train_data_x_file)
last_idx = list(range(train_x.shape[0]))
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     pca_test
   Description:   ...
   Author:        cqh
   date:          2022/7/5 21:04
-------------------------------------------------
   Change Activity:
                  2022/7/5:
-------------------------------------------------
"""
__author__ = 'cqh'

import time

from sklearn.decomposition import PCA
from utils_api import get_train_test_data

train_data, test_data = get_train_test_data()

test_data_x = test_data.drop(['Label', 'ID'], axis=1)
train_data_x = train_data.drop(['Label', 'ID'], axis=1)

# 稀疏度
spar_ratio = (train_data_x == 0).sum(axis=1).sum(axis=0) / (train_data_x.shape[0] * train_data_x.shape[1])
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
from sklearn.linear_model import LogisticRegression

train_data, _ = get_train_test_data()
train_data = train_data.sample(frac=0.1, random_state=2022)

train_data_x = train_data.drop(['ID', 'Label'], axis=1)
train_data_y = train_data['Label']

print("shape:", train_data.shape)

# LR train
lr_local = LogisticRegression(solver='liblinear', max_iter=100, n_jobs=1)

# XGB train


# 稀疏度
# spar_ratio = (train_data_x == 0).sum(axis=1).sum(axis=0) / (train_data_x.shape[0] * train_data_x.shape[1])

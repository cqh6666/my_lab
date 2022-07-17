# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     S05_lle_test
   Description:   ...
   Author:        cqh
   date:          2022/7/12 14:38
-------------------------------------------------
   Change Activity:
                  2022/7/12:
-------------------------------------------------
"""
__author__ = 'cqh'

import sys
import time

from sklearn.manifold import LocallyLinearEmbedding as LLE
from utils_api import get_train_test_data, covert_time_format

test_select = 0.1
# 获取数据
train_data, test_data = get_train_test_data()

# 处理train_data
train_data.set_index(["ID"], inplace=True)
train_data_y = train_data['Label']
train_data_x = train_data.drop(['Label'], axis=1)
# 处理test_data
test_data.set_index(["ID"], inplace=True)
test_data = test_data.sample(frac=test_select)
test_data_y = test_data['Label']
test_data_x = test_data.drop(['Label'], axis=1)


def lle_fit(n_components, n_neighbors):
    s_t = time.time()
    lle = LLE(n_components=n_components, n_neighbors=n_neighbors)
    lle.fit_transform(train_data_x)
    lle.transform(test_data_x)
    e_t = time.time()
    print("fit_time:", covert_time_format(e_t - s_t))


n_component = int(sys.argv[1])
n_neigh = int(sys.argv[2])
print("components:", n_component, "neighbors:", n_neigh)
lle_fit(n_component, n_neigh)

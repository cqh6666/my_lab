# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     S0_get_data_info
   Description:   ...
   Author:        cqh
   date:          2022/7/16 20:16
-------------------------------------------------
   Change Activity:
                  2022/7/16:
-------------------------------------------------
"""
__author__ = 'cqh'

import pandas as pd

from utils_api import get_train_test_data


train_data, test_data = get_train_test_data()

train_y = train_data['Label']
test_y = test_data['Label']

print(train_y.shape, test_y.shape)
a = train_y.value_counts(normalize=True)
print("======== train data ============")
print(a)

b = test_y.value_counts(normalize=True)
print("============ test data ========")
print(b)



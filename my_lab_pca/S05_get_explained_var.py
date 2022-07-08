# -*- coding: gbk -*-
"""
-------------------------------------------------
   File Name:     S04_patient_comp
   Description:   比较pca前后患者的相似程度
   Author:        cqh
   date:          2022/7/7 12:52
-------------------------------------------------
   Change Activity:
                  2022/7/7:
-------------------------------------------------
"""
__author__ = 'cqh'

from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

from utils_api import get_train_test_data

train_data, test_data = get_train_test_data()

test_data_x = test_data.drop(['Label', 'ID'], axis=1)
train_data_x = train_data.drop(['Label', 'ID'], axis=1)

pca = PCA(n_components=0.999)
pca.fit(test_data_x)

ev_r = pca.explained_variance_ratio_
ev_r_sum = np.cumsum(pca.explained_variance_ratio_)
pd.DataFrame({"ev_r": ev_r, "ev_r_sum": ev_r_sum}).to_csv("./result/S05_ev_r.csv")

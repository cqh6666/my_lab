# -*- coding: gbk -*-
"""
-------------------------------------------------
   File Name:     S04_patient_comp
   Description:   �Ƚ�pcaǰ���ߵ����Ƴ̶�
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

train_data_x = train_data.drop(['Label', 'ID'], axis=1)


pca = PCA(n_components=train_data_x.shape[1] - 1)
pca.fit(train_data_x)

ev_r = pca.explained_variance_ratio_
ev_r_sum = np.cumsum(pca.explained_variance_ratio_)
pd.DataFrame({"ev_r": ev_r, "ev_r_sum": ev_r_sum}).to_csv("./result/S05_get_explained_vr.csv")

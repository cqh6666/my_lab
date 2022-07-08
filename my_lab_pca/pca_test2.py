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
import numpy as np

from sklearn.decomposition import PCA
from utils_api import get_train_test_data, covert_time_format

train_data, test_data = get_train_test_data()

test_data_x = test_data.drop(['Label', 'ID'], axis=1)
train_data_x = train_data.drop(['Label', 'ID'], axis=1)


def print_pca_info(pca):
    print("pca.svd_solver:", pca.svd_solver)
    print("pca.n_components_:", pca.n_components_)
    # print("pca.explained_variance_ratio:", pca.explained_variance_ratio_)
    # print("pca.explained_variance_:", pca.explained_variance_)


def pca_params_comp():
    """
    比较不同components数值对应不同的消耗时间
    :return:
    """
    svd_solver_list = ['auto', 'full', 'arpack', 'randomized']

    for svd_solver in svd_solver_list:
        start_time = time.time()
        pca = PCA(n_components='mle', svd_solver=svd_solver, random_state=2022)
        new_test_data = pca.fit_transform(test_data_x)
        fit_tra_time = time.time()
        pca.transform(train_data_x)
        tra_time = time.time()

        # info show
        print_pca_info(pca)
        print("shape: ", test_data_x.shape, new_test_data.shape)
        print("fit_time: ", covert_time_format(fit_tra_time - start_time))
        print("tra_time: ", covert_time_format(tra_time - fit_tra_time))
        print("==========================================")

    print("done!")

pca_params_comp()


# start_time = time.time()
# pca_model = PCA(n_components='mle')
# new_test_data = pca_model.fit_transform(test_data_x) # 4:38
# fit_time = time.time()
# print("test_data_shape:", test_data_x.shape)
# print("new_test_data_shape:", new_test_data.shape)
# print("fit_cost_time:", covert_time_format(fit_time - start_time))
# print("=============================================")
# new_train_data = pca_model.transform(train_data_x) # 20s
# transform_time = time.time()
# print("train_data_shape:", train_data_x.shape)
# print("new_train_data_shape:", new_train_data.shape)
# print("transform_cost_time:", covert_time_format(transform_time - fit_time))
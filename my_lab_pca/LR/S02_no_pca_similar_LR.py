# -*- coding: gbk -*-
"""
-------------------------------------------------
   File Name:     pca_similar
   Description:   没做PCA处理，使用初始相似性度量匹配相似样本，进行计算AUC
   Author:        cqh
   date:          2022/7/5 10:07
-------------------------------------------------
   Change Activity:
                  2022/7/5:
-------------------------------------------------
"""
__author__ = 'cqh'

import os
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED

import pickle
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from utils_api import get_train_test_data, covert_time_format
from my_logger import MyLog

warnings.filterwarnings('ignore')


def get_similar_rank(pre_data_select):
    """
    选择前10%的样本，并且根据相似得到样本权重
    :param pre_data_select:
    :return:
    """
    try:
        similar_rank = pd.DataFrame(index=train_data_x.index)
        similar_rank['distance'] = cal_all_distance_by_pre_data_select(pre_data_select)
        similar_rank.sort_values('distance', inplace=True)
        patient_ids = similar_rank.index[:len_split].values

        sample_ki = similar_rank.iloc[:len_split, 0].values
        sample_ki = [(sample_ki[0] + m_sample_weight) / (val + m_sample_weight) for val in sample_ki]
    except Exception as err:
        raise err

    return patient_ids, sample_ki


def cal_all_distance_by_pre_data_select(pre_data_select):
    all_sub_cost = abs((train_data_x - pre_data_select.values) * init_similar_weight).sum(axis=1)
    return all_sub_cost


def lr_train(fit_train_x, fit_train_y, pre_data_select, sample_ki):
    lr_local = LogisticRegression(solver="liblinear", n_jobs=1, max_iter=local_lr_iter)
    lr_local.fit(fit_train_x, fit_train_y, sample_ki)
    predict_prob = lr_local.predict_proba(pre_data_select)[0][1]
    return predict_prob


def personalized_modeling(test_id, pre_data_select):
    """
    根据距离得到 某个目标测试样本对每个训练样本的距离
    test_id - patient id
    pre_data_select - dataframe
    :return: 最终的相似样本
    """
    start_time = time.time()
    patient_ids, sample_ki = get_similar_rank(pre_data_select)

    # print("patient_id: ", patient_ids[:5])
    # print("distance: ", sample_ki[:5])

    fit_train_x = train_data_x.loc[patient_ids]
    fit_train_y = train_data_y.loc[patient_ids]

    predict_prob = lr_train(fit_train_x, fit_train_y, pre_data_select, sample_ki)

    global_lock.acquire()
    test_result.loc[test_id, 'prob'] = predict_prob
    test_similar_patient_ids[test_id] = patient_ids
    global_lock.release()

    end_time = time.time()
    my_logger.info(f"patient id:{test_id} | cost_time:{covert_time_format(end_time - start_time)}...")


if __name__ == '__main__':

    my_logger = MyLog().logger

    pre_hour = 24
    root_dir = f"{pre_hour}h"
    DATA_SOURCE_PATH = f"/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/data/{root_dir}/"  # 训练集的X和Y
    MODEL_SAVE_PATH = f'/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/result/psm_with_xgb/{pre_hour}h/global_model/'

    pool_nums = 25
    test_select = 100
    select_ratio = 0.1
    m_sample_weight = 0.01

    local_lr_iter = 100

    xgb_boost_num = 50
    xgb_thread_num = 1

    my_logger.warning(
        f"[params] - model_select:LR, pool_nums:{pool_nums}, test_select:{test_select}")

    init_similar_weight_file = os.path.join(MODEL_SAVE_PATH, f'0007_{pre_hour}h_global_xgb_feature_weight_boost500.csv')
    init_similar_weight = pd.read_csv(init_similar_weight_file).squeeze().tolist()

    # 获取数据
    train_data, test_data = get_train_test_data()
    # 处理数据
    train_data.set_index(["ID"], inplace=True)

    test_data = test_data.sample(n=test_select, random_state=2022)
    test_data.set_index(["ID"], inplace=True)

    my_logger.warning(f"load_data: {train_data.shape}, {test_data.shape}")

    # 处理train_data
    train_data_y = train_data['Label']
    train_data_x = train_data.drop(['Label'], axis=1)
    # 处理test_data
    test_data_y = test_data['Label']
    test_data_x = test_data.drop(['Label'], axis=1)

    # pca降维
    pca = PCA()
    # 抽1000个患者
    len_split = int(select_ratio * train_data.shape[0])
    test_id_list = test_data_x.index.values

    test_result = pd.DataFrame(index=test_id_list, columns=['real', 'prob'])
    test_result['real'] = test_data_y

    test_similar_patient_ids = {}

    global_lock = threading.Lock()
    my_logger.warning("starting ...")
    s_t = time.time()
    # 匹配相似样本（从训练集） XGB建模 多线程
    with ThreadPoolExecutor(max_workers=pool_nums) as executor:
        thread_list = []
        for test_id in test_id_list:
            pre_data_select = test_data_x.loc[[test_id]]
            thread = executor.submit(personalized_modeling, test_id, pre_data_select)
            thread_list.append(thread)
        wait(thread_list, return_when=ALL_COMPLETED)

    e_t = time.time()
    my_logger.warning(f"done - cost_time: {covert_time_format(e_t - s_t)}...")

    # save test_similar_patient_ids
    with open(f'./result/S02_test_similar_patient_ids_LR_v2.pkl', 'wb') as file:
        pickle.dump(test_similar_patient_ids, file)

    # save result csv
    test_result.to_csv(f"./result/S02_test_result_LR_v2.csv")
    y_test, y_pred = test_result['real'], test_result['prob']
    score = roc_auc_score(y_test, y_pred)
    my_logger.info(f"personalized auc is: {score}")

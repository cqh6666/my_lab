# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     S05_kth_avg
   Description:   ...
   Author:        cqh
   date:          2022/7/14 9:22
-------------------------------------------------
   Change Activity:
                  2022/7/14:
-------------------------------------------------
"""
__author__ = 'cqh'

import os
import sys
from threading import Lock
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances

import pandas as pd
from sklearn.metrics import roc_auc_score

from utils_api import get_train_test_data, covert_time_format
from my_logger import MyLog
from xgb_utils_api import get_xgb_model_pkl, get_local_xgb_para, get_init_similar_weight

warnings.filterwarnings('ignore')


def get_similar_rank(pca_pre_data_select):
    """
    选择前10%的样本，并且根据相似得到样本权重
    :param pca_pre_data_select:
    :return:
    """
    try:
        similar_rank = pd.DataFrame(index=train_data_x.index)
        similar_rank['distance'] = abs((pca_train_data_x - pca_pre_data_select.values)).sum(axis=1)
        similar_rank.sort_values('distance', inplace=True)
        patient_ids = similar_rank.index[:len_split].values

        sample_ki = similar_rank.iloc[:len_split, 0].values
        sample_ki = [(sample_ki[0] + m_sample_weight) / (val + m_sample_weight) for val in sample_ki]
    except Exception as err:
        raise err

    return patient_ids, sample_ki


def xgb_train(fit_train_x, fit_train_y, pre_data_select, sample_ki):
    d_train_local = xgb.DMatrix(fit_train_x, label=fit_train_y, weight=sample_ki)

    xgb_local = xgb.train(params=params,
                          dtrain=d_train_local,
                          num_boost_round=num_boost_round,
                          verbose_eval=False,
                          xgb_model=xgb_model)
    d_test_local = xgb.DMatrix(pre_data_select)
    predict_prob = xgb_local.predict(d_test_local)[0]
    return predict_prob


def personalized_modeling(test_id, pre_data_select, pca_pre_data_select):
    """
    根据距离得到 某个目标测试样本对每个训练样本的距离
    test_id - patient id
    pre_data_select - dataframe
    :return: 最终的相似样本
    """
    patient_ids, sample_ki = get_similar_rank(pca_pre_data_select)

    fit_train_x = train_data_x.loc[patient_ids]
    fit_train_y = train_data_y.loc[patient_ids]
    predict_prob = xgb_train(fit_train_x, fit_train_y, pre_data_select, sample_ki)
    global_lock.acquire()
    test_result.loc[test_id, 'prob'] = predict_prob
    # test_similar_patient_ids[test_id] = patient_ids
    global_lock.release()


def trans_mean_value(test_data_x, top_k_mean):
    test_data_row = test_data_x.shape[0]
    new_test_data_x = pd.DataFrame(index=range(test_data_row), columns=test_data_x.columns)

    distance_matrix = pd.DataFrame(euclidean_distances(test_data_x))

    for i in range(test_data_row):
        top_index = distance_matrix.iloc[i].nsmallest(top_k_mean).index
        new_test_data_x.iloc[i] = test_data_x.iloc[top_index].mean(axis=0)

    new_test_data_x.index = test_data_x.index
    return new_test_data_x


def pca_reduction(train_x, test_x, n_components):
    my_logger.warning(f"starting pca by train_data...")
    # pca降维
    pca_model = PCA(n_components=n_components, random_state=2022)
    # 转换需要 * 相似性度量
    new_train_data_x = pca_model.fit_transform(train_x * init_similar_weight)
    new_test_data_x = pca_model.transform(test_x * init_similar_weight)
    # 转成df格式
    pca_train_x = pd.DataFrame(data=new_train_data_x, index=train_x.index)
    pca_test_x = pd.DataFrame(data=new_test_data_x, index=test_x.index)

    my_logger.info(f"n_components: {pca_model.n_components}, svd_solver:{pca_model.svd_solver}.")

    return pca_train_x, pca_test_x


if __name__ == '__main__':

    my_logger = MyLog().logger

    pool_nums = 20
    test_select = 1000
    select_ratio = 0.1
    m_sample_weight = 0.01

    xgb_thread_num = 1
    xgb_boost_num = 50

    is_transfer = int(sys.argv[1])
    n_components = int(sys.argv[2])
    top_k_mean = int(sys.argv[3])

    transfer_flag = "transfer" if is_transfer == 1 else "no_transfer"

    params, num_boost_round = get_local_xgb_para(xgb_thread_num=xgb_thread_num, num_boost_round=xgb_boost_num)
    xgb_model = get_xgb_model_pkl(is_transfer)
    init_similar_weight = get_init_similar_weight()

    version = 2
    # ================== save file name ====================
    patient_ids_list_file_name = f"./result/S05_test_similar_patient_ids_XGB_{transfer_flag}v{version}.pkl"
    all_result_file_name = f"./result/S05_kth_mean_pca_all_result_{transfer_flag}_{n_components}_v{version}.csv"
    # =====================================================

    my_logger.warning(
        f"[params] - version:{version}, top_k_mean:{top_k_mean}, model_select:XGB, transfer_flag:{transfer_flag}, pool_nums:{pool_nums}, test_select:{test_select}")

    # 获取数据
    train_data, test_data = get_train_test_data()

    train_data.set_index(["ID"], inplace=True)
    train_data_y = train_data['Label']
    train_data_x = train_data.drop(['Label'], axis=1)

    test_data.set_index(["ID"], inplace=True)
    test_data = test_data.sample(n=test_select)
    test_data_y = test_data['Label']
    test_data_x = test_data.drop(['Label'], axis=1)

    # 选top k 个患者并进行均值化
    mean_test_data_x = trans_mean_value(test_data_x, top_k_mean)

    my_logger.warning(f"load_data: {train_data_x.shape}, {mean_test_data_x.shape}")

    # PCA降维
    pca_train_data_x, pca_test_data_x = train_data_x, mean_test_data_x

    # 抽1000个患者
    len_split = int(select_ratio * train_data.shape[0])
    test_id_list = pca_test_data_x.index.values

    test_result = pd.DataFrame(index=test_id_list, columns=['real', 'prob'])
    test_result['real'] = test_data_y

    test_similar_patient_ids = {}

    global_lock = Lock()
    my_logger.warning("starting ...")

    s_t = time.time()
    # 匹配相似样本（从训练集） XGB建模 多线程
    with ThreadPoolExecutor(max_workers=pool_nums) as executor:
        thread_list = []
        for test_id in test_id_list:
            pre_data_select = test_data_x.loc[[test_id]]
            pca_pre_data_select = pca_test_data_x.loc[[test_id]]
            thread = executor.submit(personalized_modeling, test_id, pre_data_select, pca_pre_data_select)
            thread_list.append(thread)
        wait(thread_list, return_when=ALL_COMPLETED)

    e_t = time.time()
    my_logger.warning(f"done - cost_time: {covert_time_format(e_t - s_t)}...")

    # save test_similar_patient_ids
    # with open(patient_ids_list_file_name, 'wb') as file:
    #     pickle.dump(test_similar_patient_ids, file)

    # save result csv
    y_test, y_pred = test_result['real'], test_result['prob']
    score = roc_auc_score(y_test, y_pred)
    my_logger.warning(f"personalized auc is: {score}")

    try:
        # 保存到统一的位置
        cur_result = [[top_k_mean, score]]
        cur_result_df = pd.DataFrame(cur_result, columns=['top_k_to_mean', 'auc'])
        if os.path.exists(all_result_file_name):
            cur_result_df.to_csv(all_result_file_name, mode='a', index=False, header=False)
        else:
            cur_result_df.to_csv(all_result_file_name, index=False, header=True)
    except Exception as err:
        print(err)
# -*- coding: gbk -*-
"""
-------------------------------------------------
   File Name:     pca_similar
   Description:    ���Լ� ����XGB���Ի���ģ
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
from threading import Lock
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
import pandas as pd
from sklearn.decomposition import PCA

import xgboost as xgb
from utils_api import covert_time_format, get_train_test_x_y, save_to_csv_by_row
from xgb_utils_api import get_init_similar_weight, get_local_xgb_para, get_xgb_model_pkl
from my_logger import MyLog

warnings.filterwarnings('ignore')


def get_similar_rank(target_pre_data_select):
    """
    ѡ��ǰ10%�����������Ҹ������Ƶõ�����Ȩ��
    :param target_pre_data_select:
    :return:
    """
    try:
        similar_rank = pd.DataFrame(index=pca_train_data_x.index)
        similar_rank['distance'] = abs(pca_train_data_x - target_pre_data_select.values).sum(axis=1)
        similar_rank.sort_values('distance', inplace=True)
        patient_ids = similar_rank.index[:len_split].values

        sample_ki = similar_rank.iloc[:len_split, 0].values
        sample_ki = [(sample_ki[0] + m_sample_weight) / (val + m_sample_weight) for val in sample_ki]
    except Exception as err:
        print(err)
        sys.exit(1)

    return patient_ids, sample_ki


def xgb_train(fit_train_x, fit_train_y, pre_data_select_, sample_ki):
    """
    xgbѵ��ģ�ͣ��õ�Ԥ�����
    :return:
    """
    d_train_local = xgb.DMatrix(fit_train_x, label=fit_train_y, weight=sample_ki)
    xgb_local = xgb.train(params=params,
                          dtrain=d_train_local,
                          num_boost_round=num_boost_round,
                          verbose_eval=False,
                          xgb_model=xgb_model)
    d_test_local = xgb.DMatrix(pre_data_select_)
    predict_prob = xgb_local.predict(d_test_local)[0]
    return predict_prob


def personalized_modeling(patient_id, pre_data_select_x, pca_pre_data_select_x):
    """
    ���Ի���ģ ���� �õ� Ŀ��������Ԥ�����
    :param pre_data_select_x: ԭʼ��������
    :param pca_pre_data_select_x:  �����Ĳ�������
    :param patient_id: Ŀ�껼�ߵ�����
    :return:
    """
    # �ҵ����ƻ���ID
    patient_ids, sample_ki = get_similar_rank(pca_pre_data_select_x)

    # ɸѡ�������ƻ��� ԭʼ����
    x_train = train_data_x.loc[patient_ids]
    y_train = train_data_y.loc[patient_ids]

    # =========================== XGBר������ ================================#
    fit_train_x = x_train
    fit_train_y = y_train
    fit_test_x = pre_data_select_x
    predict_prob = xgb_train(fit_train_x, fit_train_y, fit_test_x, sample_ki)
    # =========================== XGBר������ ================================#

    try:
        global_lock.acquire()
        test_result.loc[patient_id, 'prob'] = predict_prob
        global_lock.release()
    except Exception as err:
        print(err)
        sys.exit(1)


def pca_reduction(train_x, test_x, similar_weight, n_comp):
    if n_comp >= train_x.shape[1]:
        n_comp = train_x.shape[1] - 1

    my_logger.warning(f"starting pca by train_data...")
    # pca��ά
    pca_model = PCA(n_components=n_comp, random_state=2022)
    # ת����Ҫ * �����Զ���
    new_train_data_x = pca_model.fit_transform(train_x * similar_weight)
    new_test_data_x = pca_model.transform(test_x * similar_weight)
    # ת��df��ʽ
    pca_train_x = pd.DataFrame(data=new_train_data_x, index=train_x.index)
    pca_test_x = pd.DataFrame(data=new_test_data_x, index=test_x.index)

    my_logger.info(f"n_components: {pca_model.n_components}, svd_solver:{pca_model.svd_solver}.")

    return pca_train_x, pca_test_x


if __name__ == '__main__':

    my_logger = MyLog().logger

    is_transfer = int(sys.argv[1])
    learned_metric_iteration = int(sys.argv[2])
    start_idx = int(sys.argv[3])
    end_idx = int(sys.argv[4])

    transfer_flag = "transfer" if is_transfer == 1 else "no_transfer"

    m_sample_weight = 0.01
    select = 10
    select_ratio = select * 0.01
    n_components = 100
    pool_nums = 30

    PSM_SAVE_PATH = f'./result/S06_temp/psm_{transfer_flag}'
    TEST_RESULT_SAVE_PATH = f"./result/S06_temp/test_{transfer_flag}"
    if not os.path.exists(PSM_SAVE_PATH):
        os.makedirs(PSM_SAVE_PATH)
    if not os.path.exists(TEST_RESULT_SAVE_PATH):
        os.makedirs(TEST_RESULT_SAVE_PATH)

    # =========================== XGBר�� ================================#
    version = 1
    pool_nums = 30
    xgb_thread_num = 1
    xgb_boost_num = 50
    params, num_boost_round = get_local_xgb_para(xgb_thread_num=xgb_thread_num, num_boost_round=xgb_boost_num)
    xgb_model = get_xgb_model_pkl(is_transfer)

    my_logger.warning(f"[params] - local:{xgb_boost_num}, thread:{xgb_thread_num}, pool_nums:{pool_nums}, version:{version:{version}}")
    # =========================== XGBר�� ================================#

    my_logger.warning(
        f"[params] - pool_nums:{pool_nums}, is_transfer:{is_transfer}, dim:{n_components}, test_idx_range:[{start_idx}, {end_idx}]")

    # ================================== save file======================================
    psm_file_name = f"S06_iter{learned_metric_iteration}_dim{n_components}_tra{is_transfer}_v{version}.csv"
    test_result_file_name = f"S07_test_iter{learned_metric_iteration}_dim{n_components}_tra{is_transfer}_v{version}.csv"
    # ================================== save file======================================

    if learned_metric_iteration == 0:
        psm_weight = get_init_similar_weight()
    else:
        psm_weight = pd.read_csv(os.path.join(PSM_SAVE_PATH, psm_file_name)).squeeze().tolist()

    # ��ȡ����
    train_data_x, train_data_y, test_data_x, test_data_y = get_train_test_x_y()

    final_idx = test_data_x.shape[0]
    end_idx = final_idx if end_idx > final_idx else end_idx  # ���ô�����ֵ

    # �����ν��и��Ի���ģ
    test_data_x = test_data_x.iloc[start_idx:end_idx]
    test_data_y = test_data_y.iloc[start_idx:end_idx]

    # PCA��ά
    pca_train_data_x, pca_test_data_x = pca_reduction(train_data_x, test_data_x, psm_weight, n_components)
    my_logger.warning("load data - train_data:{}, test_data:{}".format(train_data_x.shape, test_data_x.shape))

    len_split = int(select_ratio * train_data_x.shape[0])
    test_id_list = pca_test_data_x.index.values

    test_result = pd.DataFrame(index=test_id_list, columns=['real', 'prob'])
    test_result['real'] = test_data_y

    global_lock = Lock()
    my_logger.warning("starting personalized modelling...")
    s_t = time.time()
    # ƥ��������������ѵ������ XGB��ģ ���߳�
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

    test_result_file = os.path.join(TEST_RESULT_SAVE_PATH, test_result_file_name)
    # save concat test_result csv
    if save_to_csv_by_row(test_result_file, test_result):
        my_logger.info("save test result prob success!")
    else:
        my_logger.info("save error...")


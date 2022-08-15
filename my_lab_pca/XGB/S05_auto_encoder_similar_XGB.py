# -*- coding: gbk -*-
"""
-------------------------------------------------
   File Name:     pca_similar
   Description:   û��PCA����ʹ�ó�ʼ�����Զ���ƥ���������������м���AUC
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
import xgboost as xgb

import pandas as pd
from sklearn.decomposition import PCA

from my_logger import MyLog
from utils_api import get_train_test_x_y, covert_time_format, save_to_csv_by_row
from xgb_utils_api import get_xgb_model_pkl, get_local_xgb_para, get_init_similar_weight

warnings.filterwarnings('ignore')


def get_similar_rank(pca_pre_data_select_):
    """
    ѡ��ǰ10%�����������Ҹ������Ƶõ�����Ȩ��
    :param pca_pre_data_select_: ��ǰ��ѡ����
    :return:
    """
    try:
        similar_rank = pd.DataFrame(index=train_data_x.index)
        similar_rank['distance'] = abs(pca_train_data_x - pca_pre_data_select_.values).sum(axis=1)
        similar_rank.sort_values('distance', inplace=True)
        patient_ids = similar_rank.index[:len_split].values

        sample_ki = similar_rank.iloc[:len_split, 0].values
        sample_ki = [(sample_ki[0] + m_sample_weight) / (val + m_sample_weight) for val in sample_ki]
    except Exception as err:
        print(err)
        sys.exit(1)

    return patient_ids, sample_ki


def xgb_train(fit_train_x, fit_train_y, pre_data_select, sample_ki):
    """
    xgbѵ��ģ�ͣ���ԭʼ���ݽ�ģ
    :param fit_train_x:
    :param fit_train_y:
    :param pre_data_select:
    :param sample_ki:
    :return:
    """
    d_train_local = xgb.DMatrix(fit_train_x, label=fit_train_y, weight=sample_ki)
    xgb_local = xgb.train(params=params,
                          dtrain=d_train_local,
                          num_boost_round=num_boost_round,
                          verbose_eval=False,
                          xgb_model=xgb_model)
    d_test_local = xgb.DMatrix(pre_data_select)
    predict_prob = xgb_local.predict(d_test_local)[0]
    return predict_prob


def personalized_modeling(test_id_, pre_data_select_, pca_pre_data_select_):
    """
    ���ݾ���õ� ĳ��Ŀ�����������ÿ��ѵ�������ľ���
    test_id - patient id
    pre_data_select - Ŀ��ԭʼ����
    pca_pre_data_select: pca��ά���Ŀ������
    :return: ���յ���������
    """
    patient_ids, sample_ki = get_similar_rank(pca_pre_data_select_)

    try:
        fit_train_x = train_data_x.loc[patient_ids]
        fit_train_y = train_data_y.loc[patient_ids]
        predict_prob = xgb_train(fit_train_x, fit_train_y, pre_data_select_, sample_ki)
        global_lock.acquire()
        test_result.loc[test_id_, 'prob'] = predict_prob
        global_lock.release()
    except Exception as err:
        print(err)
        sys.exit(1)


if __name__ == '__main__':

    my_logger = MyLog().logger

    pool_nums = 30
    select_ratio = 0.1
    m_sample_weight = 0.01

    xgb_boost_num = 50
    xgb_thread_num = 1

    is_transfer = int(sys.argv[1])
    # �ֳ�5����ÿһ��2000����1w����������
    start_idx = int(sys.argv[2])
    end_idx = int(sys.argv[3])

    transfer_flag = "transfer" if is_transfer == 1 else "no_transfer"

    params, num_boost_round = get_local_xgb_para(xgb_thread_num=xgb_thread_num, num_boost_round=xgb_boost_num)
    xgb_model = get_xgb_model_pkl(is_transfer)
    init_similar_weight = get_init_similar_weight()

    """
    version = 1 autoEncoder 100

    """
    version = 1
    # ================== save file name ====================
    test_result_file_name = f"./result/S05_auto_encoder_xgb_test_tra{is_transfer}_v{version}.csv"
    # =====================================================

    # ��ȡ����
    train_data_x, train_data_y, test_data_x, test_data_y = get_train_test_x_y()

    final_idx = test_data_x.shape[0]
    end_idx = final_idx if end_idx > final_idx else end_idx  # ���ô�����ֵ

    # �����ν��и��Ի���ģ
    test_data_x = test_data_x.iloc[start_idx:end_idx]
    test_data_y = test_data_y.iloc[start_idx:end_idx]

    # ===========================================================
    # autoEncoder ��ά v1 100ά��
    encoder_path = "/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/code_pca/result/new_data/"
    encoder_train_data_x = pd.read_csv(os.path.join(encoder_path, "train_data_v1.csv"))
    encoder_test_data_x = pd.read_csv(os.path.join(encoder_path, "test_data_1.csv")).iloc[start_idx:end_idx]
    my_logger.warning(f"load encoder data {encoder_train_data_x.shape}, {encoder_test_data_x.shape}")
    # ==========================================================
    pca_train_data_x, pca_test_data_x = encoder_train_data_x, encoder_test_data_x

    my_logger.warning(
        f"[params] - version:{version}, transfer_flag:{transfer_flag}, pool_nums:{pool_nums}, "
        f"test_idx:[{start_idx}, {end_idx}]")

    # 10%ƥ�仼��
    len_split = int(select_ratio * train_data_x.shape[0])
    test_id_list = pca_test_data_x.index.values

    test_result = pd.DataFrame(index=test_id_list, columns=['real', 'prob'])
    test_result['real'] = test_data_y

    global_lock = threading.Lock()
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

    # save concat test_result csv
    if save_to_csv_by_row(test_result_file_name, test_result):
        my_logger.info("save test result prob success!")
    else:
        my_logger.info("save error...")

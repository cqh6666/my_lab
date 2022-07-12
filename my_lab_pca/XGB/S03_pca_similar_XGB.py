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

import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
import xgboost as xgb

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

from utils_api import get_train_test_data, covert_time_format
from xgb_utils_api import get_local_xgb_para, get_xgb_model_pkl, get_init_similar_weight

from my_logger import MyLog

warnings.filterwarnings('ignore')


def get_similar_rank(pca_pre_data_select):
    """
    ѡ��ǰ10%�����������Ҹ������Ƶõ�����Ȩ��
    :param pca_pre_data_select:
    :return:
    """
    try:
        similar_rank = pd.DataFrame(index=train_data_x.index)
        # get distance
        similar_rank['distance'] = abs(pca_train_data_x - pca_pre_data_select.values).sum(axis=1)
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
    ���ݾ���õ� ĳ��Ŀ�����������ÿ��ѵ�������ľ���
    test_id - patient id
    pre_data_select - dataframe
    :return: ���յ���������
    """
    start_time = time.time()
    patient_ids, sample_ki = get_similar_rank(pca_pre_data_select)

    fit_train_x = train_data_x.loc[patient_ids]
    fit_train_y = train_data_y.loc[patient_ids]
    predict_prob = xgb_train(fit_train_x, fit_train_y, pre_data_select, sample_ki)

    global_lock.acquire()
    test_result.loc[test_id, 'prob'] = predict_prob
    # test_similar_patient_ids[test_id] = patient_ids
    global_lock.release()

    end_time = time.time()
    # my_logger.info(f"patient id:{test_id} | cost_time:{covert_time_format(end_time - start_time)}...")


if __name__ == '__main__':

    my_logger = MyLog().logger

    pool_nums = 30
    test_select = 1000
    select_ratio = 0.1
    m_sample_weight = 0.01

    xgb_boost_num = 50
    xgb_thread_num = 1

    n_components = int(sys.argv[2])

    is_transfer = int(sys.argv[1])
    transfer_flag = "transfer" if is_transfer == 1 else "no_transfer"

    params, num_boost_round = get_local_xgb_para(xgb_thread_num=xgb_thread_num, num_boost_round=xgb_boost_num)
    xgb_model = get_xgb_model_pkl(is_transfer)
    init_similar_weight = get_init_similar_weight()

    version = 1
    # ================== save file name ====================
    patient_ids_list_file_name = f"./result/S03_test_similar_patient_ids_XGB_{transfer_flag}v{version}.pkl"
    test_result_file_name = f"./result/S03_test_result_XGB_{transfer_flag}_v{version}.csv"
    # =====================================================

    my_logger.warning(
        f"[params] - version:{version}, model_select:XGB, transfer_flag:{transfer_flag}, pool_nums:{pool_nums}, test_select:{test_select}")

    # ��ȡ����
    train_data, test_data = get_train_test_data()
    # ����train_data
    train_data.set_index(["ID"], inplace=True)
    train_data_y = train_data['Label']
    train_data_x = train_data.drop(['Label'], axis=1)
    # ����test_data
    test_data.set_index(["ID"], inplace=True)
    test_data = test_data.sample(n=test_select)
    test_data_y = test_data['Label']
    test_data_x = test_data.drop(['Label'], axis=1)

    my_logger.warning(f"train_data:{train_data.shape}, test_data:{test_data.shape}")

    my_logger.warning(f"starting pca by train_data...")
    # pca��ά
    pca_model = PCA(n_components=n_components, random_state=2022)
    # ת����Ҫ * �����Զ���
    new_train_data_x = pca_model.fit_transform(train_data_x * init_similar_weight)
    new_test_data_x = pca_model.transform(test_data_x * init_similar_weight)

    # ת��df��ʽ
    pca_train_data_x = pd.DataFrame(data=new_train_data_x, index=train_data_x.index)
    pca_test_data_x = pd.DataFrame(data=new_test_data_x, index=test_data_x.index)

    del new_train_data_x, new_test_data_x

    my_logger.info(f"n_components: {pca_model.n_components}, svd_solver:{pca_model.svd_solver}.")

    len_split = int(select_ratio * train_data.shape[0])
    test_id_list = pca_test_data_x.index.values

    test_result = pd.DataFrame(index=test_id_list, columns=['real', 'prob'])
    test_result['real'] = test_data_y

    test_similar_patient_ids = {}

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

    # save test_similar_patient_ids
    # with open(patient_ids_list_file_name, 'wb') as file:
    #     pickle.dump(test_similar_patient_ids, file)

    # save test result
    test_result.to_csv(test_result_file_name)
    y_test, y_pred = test_result['real'], test_result['prob']
    score = roc_auc_score(y_test, y_pred)
    my_logger.warning(f"personalized auc is: {score}")

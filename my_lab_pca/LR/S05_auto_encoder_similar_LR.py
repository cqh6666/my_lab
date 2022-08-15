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
from threading import Lock
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from utils_api import covert_time_format, get_train_test_x_y, save_to_csv_by_row, get_shap_value
from lr_utils_api import get_transfer_weight, get_init_similar_weight
from my_logger import MyLog
from xgb_utils_api import get_xgb_init_similar_weight

warnings.filterwarnings('ignore')


def get_similar_rank(target_pre_data_select):
    """
    选择前10%的样本，并且根据相似得到样本权重
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
        raise err

    return patient_ids, sample_ki


def lr_train(fit_train_x, fit_train_y, pre_data_select, sample_ki):
    lr_local = LogisticRegression(solver="liblinear", n_jobs=1, max_iter=local_lr_iter)
    lr_local.fit(fit_train_x, fit_train_y, sample_ki)
    predict_prob = lr_local.predict_proba(pre_data_select)[0][1]
    return predict_prob


def fit_train_test_data(patient_ids, pre_data_select_x):
    select_train_x = train_data_x.loc[patient_ids]
    if is_transfer == 1:
        transfer_weight = global_feature_weight
        fit_train_x = select_train_x * transfer_weight
        fit_test_x = pre_data_select_x * transfer_weight
    else:
        fit_train_x = select_train_x
        fit_test_x = pre_data_select_x
    return fit_test_x, fit_train_x


def personalized_modeling(patient_id, pre_data_select_x, pca_pre_data_select_x):
    """
    根据距离得到 某个目标测试样本对每个训练样本的距离
    :param pre_data_select_x: 原始测试样本
    :param pca_pre_data_select_x:  处理后的测试样本
    :param patient_id:
    :return:
    """
    try:
        patient_ids, sample_ki = get_similar_rank(pca_pre_data_select_x)
        fit_train_y = train_data_y.loc[patient_ids]
        fit_test_x, fit_train_x = fit_train_test_data(patient_ids, pre_data_select_x)
        predict_prob = lr_train(fit_train_x, fit_train_y, fit_test_x, sample_ki)
        global_lock.acquire()
        test_result.loc[patient_id, 'prob'] = predict_prob
        global_lock.release()
    except Exception as err:
        print(err)
        sys.exit(1)


if __name__ == '__main__':

    my_logger = MyLog().logger

    pool_nums = 30
    m_sample_weight = 0.01
    local_lr_iter = 100
    select = 10

    is_transfer = int(sys.argv[1])  # 0 1
    start_idx = int(sys.argv[2])
    end_idx = int(sys.argv[3])

    select_ratio = select * 0.01

    transfer_flag = "transfer" if is_transfer == 1 else "no_transfer"
    global_feature_weight = get_transfer_weight(is_transfer)

    init_similar_weight = get_init_similar_weight()
    """
    version=1 autoEncoder 100
    """
    version = 1
    # ================== save file name ====================
    test_result_file_name = f"./result/S05_auto_encoder_lr_test_tra{is_transfer}_v{version}.csv"
    # =====================================================

    # 获取数据
    train_data_x, train_data_y, test_data_x, test_data_y = get_train_test_x_y()

    final_idx = test_data_x.shape[0]
    end_idx = final_idx if end_idx > final_idx else end_idx  # 不得大过最大值

    # 分批次进行个性化建模
    test_data_x = test_data_x.iloc[start_idx:end_idx]
    test_data_y = test_data_y.iloc[start_idx:end_idx]

    # ===========================================================
    # autoEncoder 降维 v1 100维度
    encoder_path = "/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/code_pca/result/new_data/"
    encoder_train_data_x = pd.read_csv(os.path.join(encoder_path, "train_data_v1.csv"))
    encoder_test_data_x = pd.read_csv(os.path.join(encoder_path, "test_data_1.csv")).iloc[start_idx:end_idx]
    my_logger.warning(f"load encoder data {encoder_train_data_x.shape}, {encoder_test_data_x.shape}")
    # ==========================================================
    pca_train_data_x, pca_test_data_x = encoder_train_data_x, encoder_test_data_x

    my_logger.warning("load data - train_data:{}, test_data:{}".format(train_data_x.shape, test_data_x.shape))
    my_logger.warning(
        f"[params] - model_select:LR, pool_nums:{pool_nums}, is_transfer:{is_transfer}, max_iter:{local_lr_iter}, select:{select}, version:{version}, test_idx:[{start_idx}, {end_idx}]")

    len_split = int(select_ratio * train_data_x.shape[0])
    test_id_list = pca_test_data_x.index.values

    test_result = pd.DataFrame(index=test_id_list, columns=['real', 'prob'])
    test_result['real'] = test_data_y

    global_lock = Lock()
    my_logger.warning("starting personalized modelling...")
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

    # save concat test_result csv
    if save_to_csv_by_row(test_result_file_name, test_result):
        my_logger.info("save test result prob success!")
    else:
        my_logger.info("save error...")


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


def get_similar_rank(pca_pre_data_select):
    """
    选择前10%的样本，并且根据相似得到样本权重
    :param pre_data_select:
    :return:
    """
    try:
        similar_rank = pd.DataFrame(index=train_data_x.index)
        similar_rank['distance'] = abs(pca_train_data_x - pca_pre_data_select.values).sum(axis=1)
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
    xgb训练模型，用原始数据建模
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


def personalized_modeling(test_id, pre_data_select, pca_pre_data_select):
    """
    根据距离得到 某个目标测试样本对每个训练样本的距离
    test_id - patient id
    pre_data_select - 目标原始样本
    pca_pre_data_select: pca降维后的目标样本
    :return: 最终的相似样本
    """
    patient_ids, sample_ki = get_similar_rank(pca_pre_data_select)

    try:
        fit_train_x = train_data_x.loc[patient_ids]
        fit_train_y = train_data_y.loc[patient_ids]
        predict_prob = xgb_train(fit_train_x, fit_train_y, pre_data_select, sample_ki)
        global_lock.acquire()
        test_result.loc[test_id, 'prob'] = predict_prob
        global_lock.release()
    except Exception as err:
        print(err)
        sys.exit(1)


def pca_reduction(train_x, test_x, similar_weight, n_comp):
    if n_comp >= train_data_x.shape[1]:
        n_comp = train_data_x.shape[1] - 1

    my_logger.warning(f"starting pca by train_data...")
    # pca降维
    pca_model = PCA(n_components=n_comp, random_state=2022)
    # 转换需要 * 相似性度量
    new_train_data_x = pca_model.fit_transform(train_x * similar_weight)
    new_test_data_x = pca_model.transform(test_x * similar_weight)
    # 转成df格式
    pca_train_x = pd.DataFrame(data=new_train_data_x, index=train_x.index)
    pca_test_x = pd.DataFrame(data=new_test_data_x, index=train_x.index)

    my_logger.info(f"n_components: {pca_model.n_components}, svd_solver:{pca_model.svd_solver}.")

    return pca_train_x, pca_test_x


if __name__ == '__main__':

    my_logger = MyLog().logger

    pool_nums = 30
    select_ratio = 0.1
    m_sample_weight = 0.01

    xgb_boost_num = 50
    xgb_thread_num = 1

    is_transfer = int(sys.argv[1])
    n_components = int(sys.argv[2])
    # 分成5批，每一批2000，共1w个测试样本
    start_idx = int(sys.argv[3])
    end_idx = int(sys.argv[4])

    transfer_flag = "transfer" if is_transfer == 1 else "no_transfer"

    params, num_boost_round = get_local_xgb_para(xgb_thread_num=xgb_thread_num, num_boost_round=xgb_boost_num)
    xgb_model = get_xgb_model_pkl(is_transfer)
    init_similar_weight = get_init_similar_weight()

    """
    version=1 xgb_boost_num=50
    version=2 xgb_boost_num=25
    version=3 xgb_boost_num=100

    """
    version = 1
    # ================== save file name ====================
    test_result_file_name = f"./result/S03_pca_xgb_test_tra{is_transfer}_comp{n_components}_v{version}.csv"
    # =====================================================

    # 获取数据
    train_data_x, train_data_y, test_data_x, test_data_y = get_train_test_x_y()

    final_idx = test_data_x.shape[0]
    end_idx = final_idx if end_idx > final_idx else end_idx  # 不得大过最大值

    # 分批次进行个性化建模
    test_data_x = test_data_x.iloc[start_idx:end_idx]
    test_data_y = test_data_y.iloc[start_idx:end_idx]

    # PCA降维
    pca_train_data_x, pca_test_data_x = pca_reduction(train_data_x, test_data_x, init_similar_weight, n_components)

    my_logger.warning(
        f"[params] - version:{version}, transfer_flag:{transfer_flag}, pool_nums:{pool_nums}, "
        f"test_idx:[{start_idx}, {end_idx}]")

    # 10%匹配患者
    len_split = int(select_ratio * train_data_x.shape[0])
    test_id_list = pca_test_data_x.index.values

    test_result = pd.DataFrame(index=test_id_list, columns=['real', 'prob'])
    test_result['real'] = test_data_y

    global_lock = threading.Lock()
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
    save_to_csv_by_row(test_result_file_name, test_result)
    my_logger.info("save test result prob success!")
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
import feather

import pandas as pd
import numpy as np

from utils_api import get_train_test_x_y, covert_time_format, save_to_csv_by_row, get_target_test_id
from my_logger import MyLog
from xgb_utils_api import get_xgb_model_pkl, get_local_xgb_para, get_init_similar_weight

warnings.filterwarnings('ignore')


def get_similar_patient_ids(test_id, pre_data_select):
    """
    根据距离得到 某个目标测试样本对每个训练样本的距离
    test_id - patient id
    pre_data_select - dataframe
    :return: 最终的相似样本
    """
    similar_rank = pd.DataFrame(index=train_data_x.index)
    similar_rank['distance'] = abs((train_data_x - pre_data_select.values) * init_similar_weight).sum(axis=1)
    similar_rank.sort_values('distance', inplace=True)
    patient_ids = similar_rank.index[:len_split].values

    global_lock.acquire()
    patient_ids_df.loc[test_id] = patient_ids
    global_lock.release()


if __name__ == '__main__':

    my_logger = MyLog().logger

    pool_nums = 30
    select_ratio = 0.1
    m_sample_weight = 0.01

    is_transfer = int(sys.argv[1])

    xgb_thread_num = 1
    xgb_boost_num = 50

    transfer_flag = "transfer" if is_transfer == 1 else "no_transfer"

    params, num_boost_round = get_local_xgb_para(xgb_thread_num=xgb_thread_num, num_boost_round=xgb_boost_num)
    xgb_model = get_xgb_model_pkl(is_transfer)
    init_similar_weight = get_init_similar_weight()

    version = 1
    # ================== save file name ====================
    test_result_file_name = f"./result/similar_patient_ids/S02_similar_patient_ids_tra{is_transfer}_v{version}.csv"
    # =====================================================

    # 获取数据
    train_data_x, train_data_y, test_data_x, test_data_y = get_train_test_x_y()

    # 选出50个正例，50个负例
    test_ids_1, test_ids_0 = get_target_test_id()
    test_ids = np.concatenate((test_ids_1, test_ids_0), axis=0)

    # 分批次进行个性化建模
    test_data_x = test_data_x.loc[test_ids]
    test_data_y = test_data_y.loc[test_ids]

    my_logger.warning("load data - train_data:{}, test_data:{}".format(train_data_x.shape, test_data_x.shape))
    my_logger.warning(
        f"[params] - version:{version}, transfer_flag:{transfer_flag}, pool_nums:{pool_nums}")

    # 10%匹配患者
    len_split = int(select_ratio * train_data_x.shape[0])
    test_id_list = test_data_x.index.values

    patient_ids_df = pd.DataFrame(index=test_ids, columns=range(1, len_split+1))

    global_lock = threading.Lock()
    my_logger.warning("starting ...")

    s_t = time.time()
    # 匹配相似样本（从训练集） XGB建模 多线程
    with ThreadPoolExecutor(max_workers=pool_nums) as executor:
        thread_list = []
        for test_id in test_id_list:
            pre_data_select = test_data_x.loc[[test_id]]
            thread = executor.submit(get_similar_patient_ids, test_id, pre_data_select)
            thread_list.append(thread)
        wait(thread_list, return_when=ALL_COMPLETED)

    e_t = time.time()
    my_logger.warning(f"done - cost_time: {covert_time_format(e_t - s_t)}...")

    # save concat test_result feather
    patient_ids_df.to_csv(test_result_file_name)
    my_logger.info("save test result prob success!")
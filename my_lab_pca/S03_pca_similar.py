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

import argparse
import os
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
import xgboost as xgb

import pickle
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from utils_api import get_train_test_data, covert_time_format
from my_logger import MyLog

warnings.filterwarnings('ignore')


def get_local_xgb_para():
    """personal xgb para"""
    params = {
        'booster': 'gbtree',
        'max_depth': 11,
        'min_child_weight': 7,
        'subsample': 1,
        'colsample_bytree': 0.7,
        'eta': 0.05,
        'objective': 'binary:logistic',
        'nthread': xgb_thread_num,
        'verbosity': 0,
        'eval_metric': 'logloss',
        'seed': 998,
        'tree_method': 'hist'
    }
    num_boost_round = xgb_boost_num
    return params, num_boost_round


def get_similar_rank(pca_pre_data_select):
    """
    选择前10%的样本，并且根据相似得到样本权重
    :param pca_pre_data_select:
    :return:
    """
    try:
        similar_rank = pd.DataFrame(index=train_data_x.index)
        similar_rank['distance'] = cal_all_distance_by_pre_data_select(pca_pre_data_select)
        similar_rank.sort_values('distance', inplace=True)
        patient_ids = similar_rank.index[:len_split].values

        sample_ki = similar_rank.iloc[:len_split, 0].values
        sample_ki = [(sample_ki[0] + m_sample_weight) / (val + m_sample_weight) for val in sample_ki]
    except Exception as err:
        raise err

    return patient_ids, sample_ki


def cal_all_distance_by_pre_data_select(pca_pre_data_select):
    """
    get distance by target patient and all train patients with similar weight
    需要保证前面有 患者ID index
    :param pca_pre_data_select: 选中的患者
    :return:
    """
    # pca 降维
    pca_train_data_x = pca_model.transform(train_data_x * init_similar_weight)
    all_distance = abs(pca_train_data_x - pca_pre_data_select.values).sum(axis=1)
    return all_distance


def xgb_train(fit_train_x, fit_train_y, pre_data_select, sample_ki):
    d_train_local = xgb.DMatrix(fit_train_x, label=fit_train_y, weight=sample_ki)
    params, num_boost_round = get_local_xgb_para()
    xgb_local = xgb.train(params=params,
                          dtrain=d_train_local,
                          num_boost_round=num_boost_round,
                          verbose_eval=False,
                          xgb_model=None)
    d_test_local = xgb.DMatrix(pre_data_select)
    predict_prob = xgb_local.predict(d_test_local)[0]
    return predict_prob


def lr_train(fit_train_x, fit_train_y, pre_data_select, sample_ki):
    lr_local = LogisticRegression(solver="liblinear", n_jobs=1, max_iter=local_lr_iter)
    lr_local.fit(fit_train_x, fit_train_y, sample_ki)
    predict_prob = lr_local.predict_proba(pre_data_select)[0][1]
    return predict_prob


def personalized_modeling(test_id, pre_data_select, pca_pre_data_select):
    """
    根据距离得到 某个目标测试样本对每个训练样本的距离
    test_id - patient id
    pre_data_select - dataframe
    :return: 最终的相似样本
    """
    start_time = time.time()
    patient_ids, sample_ki = get_similar_rank(pca_pre_data_select)

    # print("patient_id: ", patient_ids[:5])
    # print("distance: ", sample_ki[:5])

    fit_train_x = train_data_x.loc[patient_ids]
    fit_train_y = train_data_y.loc[patient_ids]

    if model_name == "XGB":
        predict_prob = xgb_train(fit_train_x, fit_train_y, pre_data_select, sample_ki)
    else:
        predict_prob = lr_train(fit_train_x, fit_train_y, pre_data_select, sample_ki)

    global_lock.acquire()
    test_result.loc[test_id, 'prob'] = predict_prob
    test_similar_patient_ids[test_id] = patient_ids
    global_lock.release()

    end_time = time.time()
    my_logger.info(f"patient id:{test_id} | cost_time:{covert_time_format(end_time - start_time)}...")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="pca - train personalized model by different models")
    parser.add_argument("--model", "-m", default=1, type=int, choices=[1, 2], help="1-XGB, 2-LR")
    args = parser.parse_args()

    my_logger = MyLog().logger

    pre_hour = 24
    root_dir = f"{pre_hour}h"
    DATA_SOURCE_PATH = f"/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/data/{root_dir}/"  # 训练集的X和Y
    MODEL_SAVE_PATH = f'/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/result/psm_with_xgb/{pre_hour}h/global_model/'

    # 1 xgb 2 lr
    model_select = args.model
    if model_select == 1:
        model_name = "XGB"
    elif model_select == 2:
        model_name = "LR"
    else:
        raise Exception('no found model - params: {}'.format(model_select))

    pool_nums = 25
    test_select = 100
    select_ratio = 0.1
    m_sample_weight = 0.01

    local_lr_iter = 100

    xgb_boost_num = 50
    xgb_thread_num = 1

    my_logger.warning(
        f"[params] - model_select:{model_name}, pool_nums:{pool_nums}, test_select:{test_select}")

    init_similar_weight_file = os.path.join(MODEL_SAVE_PATH, f'0007_{pre_hour}h_global_xgb_feature_weight_boost500.csv')
    init_similar_weight = pd.read_csv(init_similar_weight_file).squeeze().tolist()

    # 获取数据
    train_data, test_data = get_train_test_data()
    # 处理数据
    train_data.set_index(["ID"], inplace=True)
    test_data.set_index(["ID"], inplace=True)

    # 处理train_data
    train_data_y = train_data['Label']
    train_data_x = train_data.drop(['Label'], axis=1)
    # 处理test_data
    test_data_x = test_data.drop(['Label'], axis=1)

    my_logger.warning(f"load_data: {train_data.shape}, {test_data.shape}")
    my_logger.warning(f"starting pca by test_data...")

    # pca降维
    n_components = train_data.shape[1] - 1
    pca_model = PCA(n_components=n_components, random_state=2022)
    pca_model.fit(test_data_x)

    # 处理test_data (筛选)
    test_data = test_data.sample(n=test_select, random_state=2022)
    test_data_y = test_data['Label']
    test_data_x = test_data.drop(['Label'], axis=1)

    pca_test_data_x = pd.DataFrame(data=pca_model.transform(test_data_x * init_similar_weight), index=test_data_x.index)
    my_logger.info(f"n_components: {pca_model.n_components}, svd_solver:{pca_model.svd_solver}.")

    # 抽1000个患者
    len_split = int(select_ratio * train_data.shape[0])
    test_id_list = pca_test_data_x.index.values

    test_result = pd.DataFrame(index=test_id_list, columns=['real', 'prob'])
    test_result['real'] = test_data_y

    test_similar_patient_ids = {}

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

    # save test_similar_patient_ids
    with open(f'./result/S03_test_similar_patient_ids_{model_name}_v2.pkl', 'wb') as file:
        pickle.dump(test_similar_patient_ids, file)

    # save result csv
    test_result.to_csv(f"./result/S03_test_result_{model_name}_v2.csv")
    y_test, y_pred = test_result['real'], test_result['prob']
    score = roc_auc_score(y_test, y_pred)
    my_logger.info(f"personalized auc is: {score}")

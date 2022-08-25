# encoding=gbk
"""
���߳���XGB����
"""
from threading import Lock
import time
import numpy as np
import pandas as pd
from random import shuffle
import warnings
import os

from sklearn.decomposition import PCA
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
import sys
from gc import collect
from my_logger import MyLog

from utils_api import get_train_test_x_y

import xgboost as xgb
from xgb_utils_api import get_local_xgb_para, get_xgb_model_pkl, get_init_similar_weight

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


def learn_similarity_measure(test_id_, pre_data_x_select_, pca_pre_data_x_select_, pre_data_y_select_):
    """
    ѧϰ�����Զ���
    :param test_id_:  Ŀ�껼�ߵ�����(0-1000)
    :param pre_data_x_select_: Ŀ�껼�ߵ������� df
    :param pca_pre_data_x_select_: pca��ά���Ŀ�껼�������� df
    :param pre_data_y_select_: Ŀ�껼�ߵı�ǩֵ
    :return:
    """
    # �ҵ����ƻ���ID
    patient_ids, sample_ki = get_similar_rank(pca_pre_data_x_select_)

    # ɸѡ�������ƻ��� ԭʼ����
    x_train = train_rank_x.loc[patient_ids]
    y_train = train_rank_y.loc[patient_ids]

    # =========================== XGBר������ ================================#
    fit_train_x = x_train
    fit_train_y = y_train
    fit_test_x = pre_data_x_select_
    predict_prob = xgb_train(fit_train_x, fit_train_y, fit_test_x, sample_ki)
    # =========================== XGBר������ ================================#

    # ÿ�������Ĳ���ľ�ֵ
    mean_r = np.mean(abs(x_train - pre_data_x_select_.values))
    # ��������ľ�ֵ
    y = abs(pre_data_y_select_ - predict_prob)

    global iteration_data
    global iteration_y
    global lock
    try:
        lock.acquire()
        iteration_data.loc[test_id_, :] = mean_r
        iteration_y[test_id_] = y
    except Exception as err:
        print(err)
        sys.exit(1)
    finally:
        lock.release()


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
    init_iteration = int(sys.argv[2])

    transfer_flag = "no_transfer" if is_transfer == 0 else "transfer"

    PSM_SAVE_PATH = f'./result/S06_temp/psm_{transfer_flag}'
    if not os.path.exists(PSM_SAVE_PATH):
        os.makedirs(PSM_SAVE_PATH)

    train_x, train_y, _, _ = get_train_test_x_y()
    data_columns = train_x.columns

    # ----- similarity learning para -----
    cur_iteration = init_iteration + 1

    l_rate = 0.00001
    select_rate = 0.1
    regularization_c = 0.05
    m_sample_weight = 0.01
    n_personal_model_each_iteration = 1000
    n_components = 100

    # =========================== XGBר�� ================================#
    version = 1
    pool_nums = 30
    step = 10
    xgb_thread_num = 1
    xgb_boost_num = 50
    params, num_boost_round = get_local_xgb_para(xgb_thread_num=xgb_thread_num, num_boost_round=xgb_boost_num)
    xgb_model = get_xgb_model_pkl(is_transfer)

    my_logger.warning(
        f"[params] - local:{xgb_boost_num}, thread:{xgb_thread_num}, step:{step}, pool_nums:{pool_nums}, version:{version}")
    # =========================== XGBר�� ================================#

    # ================================== save file======================================
    psm_file_name = "S06_iter{}_dim{}_tra{}_v{}.csv"
    # ================================== save file======================================

    my_logger.warning(f"[params] - is_transfer:{is_transfer}, init_iter:{init_iteration}, dim:{n_components}")
    # ----- �����Զ��� -----
    if init_iteration == 0:
        psm_weight = get_init_similar_weight()
    else:
        psm_file = os.path.join(PSM_SAVE_PATH, psm_file_name.format(init_iteration, n_components, is_transfer, version))
        psm_weight = pd.read_csv(psm_file).squeeze().tolist()
    my_logger.warning(f"load psm_weight: {init_iteration}")

    lock = Lock()
    my_logger.warning("start iteration ... ")

    # ----- iteration -----
    # one iteration includes 1000 personal models
    for iteration_idx in range(cur_iteration, cur_iteration + step):
        last_idx = list(range(train_x.shape[0]))
        # ����˳��
        shuffle(last_idx)
        last_x = train_x.loc[last_idx, :]
        last_x.reset_index(drop=True, inplace=True)
        last_y = train_y.loc[last_idx]
        last_y.reset_index(drop=True, inplace=True)

        # ѡȡ1000��������ΪĿ�껼��
        select_x = last_x.loc[:n_personal_model_each_iteration - 1, :].copy()
        select_x.reset_index(drop=True, inplace=True)
        select_y = last_y.loc[:n_personal_model_each_iteration - 1].copy()
        select_y.reset_index(drop=True, inplace=True)

        # ��Ϊѵ��������ƥ��������ƥ�����ƻ��ߵ���������
        train_rank_x = last_x.loc[n_personal_model_each_iteration - 1:, :].copy()
        train_rank_x.reset_index(drop=True, inplace=True)
        train_rank_y = last_y.loc[n_personal_model_each_iteration - 1:].copy()
        train_rank_y.reset_index(drop=True, inplace=True)

        # pca ��ά
        pca_train_data_x, pca_test_data_x = pca_reduction(train_rank_x, select_x, psm_weight, n_components)

        # ѡȡ������������������С 10%
        len_split = int(train_rank_x.shape[0] * select_rate)

        # ��������֮��Ĳ��ƽ��ֵ �� ����Ԥ��y
        personal_list = range(n_personal_model_each_iteration)  # Ҳ����test_id_list
        iteration_data = pd.DataFrame(index=personal_list, columns=data_columns)
        iteration_y = pd.Series(index=personal_list)

        pg_start_time = time.time()

        with ThreadPoolExecutor(max_workers=pool_nums) as executor:
            thread_list = []
            for test_id in personal_list:
                # ԭʼ���ݵ� Ŀ�껼��
                pre_data_x_select = select_x.loc[[test_id]]
                # pca��ά��� Ŀ�껼��
                pca_pre_data_x_select = pca_test_data_x.loc[[test_id]]
                pre_data_y_select = select_y.loc[test_id]  # target y label

                thread = executor.submit(learn_similarity_measure, test_id, pre_data_x_select, pca_pre_data_x_select,
                                         pre_data_y_select)
                thread_list.append(thread)
            wait(thread_list, return_when=ALL_COMPLETED)

        run_time = round(time.time() - pg_start_time, 2)
        my_logger.warning(
            f"iter idx:{iteration_idx} | build {n_personal_model_each_iteration} models need: {run_time} s")

        # ----- update normalize weight -----
        # 1000 * columns     columns
        """
        iteration_data ����Ŀ��������ǰN������������ÿ�������Ĳ���ƽ��ֵ
        ����normaliza_weight ���� ����ÿ�������в�һ������Ҫ�ԣ���Ҫ�Ըߵ���������ͻ����
        all_error ���Ǽ����������������Ȩֵ֮��
        """
        new_similar = iteration_data * psm_weight
        all_error = new_similar.sum(axis=1)

        new_ki = []
        risk_gap = [real - pred for real, pred in zip(list(iteration_y), list(all_error))]
        # ���е��л��е����ݱ�SqueezeΪһ��Series��
        for idx, value in enumerate(psm_weight):
            features_x = list(iteration_data.iloc[:, idx])
            plus_list = [a * b for a, b in zip(risk_gap, features_x)]
            new_value = value + l_rate * (sum(plus_list) - regularization_c * value)
            new_ki.append(new_value)

        psm_weight = list(map(lambda x: x if x > 0 else 0, new_ki))
        # list -> dataframe
        psm_weight_df = pd.DataFrame({f'psm_update_{iteration_idx}': psm_weight})

        try:
            if iteration_idx % 5 == 0:
                wi_file = os.path.join(PSM_SAVE_PATH,
                                       psm_file_name.format(iteration_idx, n_components, is_transfer, version))
                psm_weight_df.to_csv(wi_file, index=False)
                my_logger.warning(f"iter idx: {iteration_idx} | save {wi_file} success!")
        except Exception as err:
            my_logger.error(f"iter idx: {iteration_idx} | save error!")

        del iteration_data, iteration_y
        collect()

        my_logger.warning(f"======================= {iteration_idx} rounds done ! ========================")

    print("run done!")

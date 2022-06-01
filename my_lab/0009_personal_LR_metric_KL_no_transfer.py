# encoding=gbk
"""
build personal xgb models for test data with learned KL metric
"""
import os
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
import threading
import pandas as pd
from gc import collect
import warnings
import sys
import time
from my_logger import MyLog
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings('ignore')


def personalized_modeling(pre_data, idx, x_test):
    """build personal model for target sample from test datasets"""

    # personalized_modeling_start_time = time.time()

    similar_rank = pd.DataFrame()

    similar_rank['data_id'] = train_x.index.tolist()
    similar_rank['Distance'] = (abs((train_x - pre_data) * feature_weight)).sum(axis=1)

    similar_rank.sort_values('Distance', inplace=True)
    similar_rank.reset_index(drop=True, inplace=True)
    select_id = similar_rank.iloc[:len_split, 0].values

    select_train_x = train_x.iloc[select_id, :]
    select_train_y = train_y.iloc[select_id]
    fit_train = select_train_x
    fit_test = x_test

    sample_ki = similar_rank.iloc[:len_split, 1].tolist()
    sample_ki = [(sample_ki[0] + m_sample_weight) / (val + m_sample_weight) for val in sample_ki]

    lr_local = LogisticRegression(solver='liblinear')
    lr_local.fit(fit_train, select_train_y, sample_ki)

    y_predict = lr_local.predict_proba(fit_test)[:, 1]
    my_logger.info(f"[{idx}] - y_predict: {y_predict}")
    global_lock.acquire()
    test_result.loc[idx, 'prob'] = y_predict
    global_lock.release()

    # run_time = round(time.time() - personalized_modeling_start_time, 2)
    # my_logger.info(f"idx:{idx} | build time:{run_time}s")


def get_feature_weight_list(metric_iter):
    """
    �õ�����Ȩ���б�
    :param metric_iter: ����������str���ͣ�
    :return:
    """
    if int(metric_iter) == 0:
        feature_importance_file = os.path.join(MODEL_SAVE_PATH, f"0006_{pre_hour}h_global_lr.csv")
    else:
        weight_file_name = f"0008_{pre_hour}h_{metric_iter}_psm_no_transfer.csv"
        feature_importance_file = os.path.join(PSM_SAVE_PATH, weight_file_name)

    f_weight = pd.read_csv(feature_importance_file)
    f_weight = f_weight.squeeze().tolist()
    return f_weight


def get_train_test_data():
    """
    ѵ�������Լ����ݻ�ȡ
    :return:
    """
    train_x = pd.read_feather(
        os.path.join(DATA_SOURCE_PATH, "all_x_train_24h_norm_dataframe_999_miss_medpx_max2dist.feather"))
    train_y = pd.read_feather(
        os.path.join(DATA_SOURCE_PATH, "all_y_train_24h_norm_dataframe_999_miss_medpx_max2dist.feather"))['Label']
    test_x = pd.read_feather(
        os.path.join(DATA_SOURCE_PATH, "all_x_test_24h_norm_dataframe_999_miss_medpx_max2dist.feather"))
    test_y = pd.read_feather(
        os.path.join(DATA_SOURCE_PATH, "all_y_test_24h_norm_dataframe_999_miss_medpx_max2dist.feather"))['Label']

    return train_x, train_y, test_x, test_y


def params_logger_info_show():
    my_logger.warning(
        f"[iter params] - learned_iter:{learned_metric_iteration}, pool_nums:{pool_nums}, start_idx:{start_idx}, end_idx:{end_idx}, transfer_flag:no_transfer")


if __name__ == '__main__':

    start_idx = int(sys.argv[1])
    end_idx = int(sys.argv[2])
    learned_metric_iteration = str(sys.argv[3])

    pre_hour = 24
    select_ratio = 0.1
    m_sample_weight = 0.01
    pool_nums = 20

    DATA_SOURCE_PATH = f"/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/data/24h/"  # ѵ������X��Y
    MODEL_SAVE_PATH = f'/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/result/personal_model_with_lr/{pre_hour}h/global_model/'
    PSM_SAVE_PATH = f'/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/result/personal_model_with_lr/{pre_hour}h/no_transfer_psm/'
    TEST_RESULT_PATH = f'/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/result/personal_model_with_lr/{pre_hour}h/test_result_no_transfer/'

    my_logger = MyLog().logger

    # ѵ������X��Y
    train_x, train_y, test_x, test_y = get_train_test_data()

    final_idx = test_x.shape[0]
    end_idx = final_idx if end_idx > final_idx else end_idx
    len_split = int(train_x.shape[0] * select_ratio)  # the number of selected train data

    # ��ȡ������k�ε������Զ���csv�ļ�
    feature_weight = get_feature_weight_list(metric_iter=learned_metric_iteration)

    # ��ʾ������Ϣ
    params_logger_info_show()

    # init test result
    test_result = pd.DataFrame(columns=['real', 'prob'])
    test_result['real'] = test_y.iloc[start_idx:end_idx]

    global_lock = threading.Lock()
    start_time = time.time()
    # init thread list
    thread_list = []
    pool = ThreadPoolExecutor(max_workers=pool_nums)
    # build personalized model for each test sample
    for test_idx in range(start_idx, end_idx):
        pre_data_select = test_x.loc[test_idx, :]
        x_test_select = test_x.loc[[test_idx], :]

        # execute multi threads
        thread = pool.submit(personalized_modeling, pre_data_select, test_idx, x_test_select)
        thread_list.append(thread)

    # wait for all threads completing
    wait(thread_list, return_when=ALL_COMPLETED)
    collect()

    run_time = round(time.time() - start_time, 2)
    my_logger.warning(f"build all model need time: {run_time} s")

    # ----- save result -----
    try:
        test_result_csv = os.path.join(TEST_RESULT_PATH, f'0009_{learned_metric_iteration}_{start_idx}_{end_idx}_prob_no_transfer.csv')
        test_result.to_csv(test_result_csv, index=False)
        my_logger.warning(f"save {test_result_csv} success!")
    except Exception as err:
        my_logger.error(err)
        raise err
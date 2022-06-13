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

from sklearn.preprocessing import StandardScaler

from my_logger import MyLog
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings('ignore')


def personalized_modeling(pre_data, idx, x_test):
    """build personal model for target sample from test datasets"""

    # personalized_modeling_start_time = time.time()

    similar_rank = pd.DataFrame()

    similar_rank['data_id'] = train_x.index.tolist()
    similar_rank['distance'] = (abs((train_x - pre_data) * psm_weight)).sum(axis=1)

    similar_rank.sort_values('distance', inplace=True)
    similar_rank.reset_index(drop=True, inplace=True)
    select_id = similar_rank.iloc[:len_split, 0].values

    select_train_x = train_x.iloc[select_id, :]
    select_train_y = train_y.iloc[select_id]

    if is_transfer == 1:
        init_weight = global_init_normalize_weight
        fit_train = select_train_x * init_weight
        fit_test = x_test * init_weight
    else:
        fit_train = select_train_x
        fit_test = x_test

    sample_ki = similar_rank.iloc[:len_split, 1].tolist()
    sample_ki = [(sample_ki[0] + m_sample_weight) / (val + m_sample_weight) for val in sample_ki]

    lr_local = LogisticRegression(solver="liblinear", n_jobs=1, max_iter=local_lr_iter)
    lr_local.fit(fit_train, select_train_y, sample_ki)

    y_predict = lr_local.predict_proba(fit_test)[:, 1]
    global_lock.acquire()
    test_result.loc[idx, 'prob'] = y_predict
    global_lock.release()

    # run_time = round(time.time() - personalized_modeling_start_time, 2)
    # my_logger.info(f"idx:{idx} | build time:{run_time}s")


def get_feature_weight_list(metric_iter):
    """
    得到特征权重列表
    :param metric_iter: 迭代次数（str类型）
    :return:
    """
    if metric_iter == 0:
        p_weight_file = os.path.join(MODEL_SAVE_PATH, f"0006_{pre_hour}h_global_lr_liblinear_{global_lr_iter}.csv")
    else:
        p_weight_file_name = f"0008_{pre_hour}h_{metric_iter}_psm_{transfer_flag}.csv"
        p_weight_file = os.path.join(PSM_SAVE_PATH, p_weight_file_name)

    p_weight = pd.read_csv(p_weight_file).squeeze().tolist()
    return p_weight


def get_train_test_data():
    """
    训练集测试集数据获取
    :return:
    """
    train_x = pd.read_feather(
        os.path.join(DATA_SOURCE_PATH, f"all_x_train_{pre_hour}_df_rm1_norm1.feather"))
    train_y = pd.read_feather(
        os.path.join(DATA_SOURCE_PATH, f"all_y_train_{pre_hour}_df_rm1_norm1.feather"))['Label']
    test_x = pd.read_feather(
        os.path.join(DATA_SOURCE_PATH, f"all_x_test_{pre_hour}_df_rm1_norm1.feather"))
    test_y = pd.read_feather(
        os.path.join(DATA_SOURCE_PATH, f"all_y_test_{pre_hour}_df_rm1_norm1.feather"))['Label']

    return train_x, train_y, test_x, test_y


if __name__ == '__main__':

    start_idx = int(sys.argv[1])
    end_idx = int(sys.argv[2])
    is_transfer = int(sys.argv[3])
    learned_metric_iteration = int(sys.argv[4])

    transfer_flag = "no_transfer" if is_transfer == 0 else "transfer"

    pre_hour = 24
    select_ratio = 0.1
    m_sample_weight = 0.01
    pool_nums = 20
    global_lr_iter = 400
    local_lr_iter = 100

    root_dir = f"{pre_hour}h_old2"
    DATA_SOURCE_PATH = f"/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/data/{root_dir}/"  # 训练集的X和Y
    MODEL_SAVE_PATH = f'/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/result/personal_model_with_lr/{root_dir}/global_model/'
    PSM_SAVE_PATH = f'/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/result/personal_model_with_lr/{root_dir}/psm_{transfer_flag}/'
    TEST_RESULT_PATH = f'/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/result/personal_model_with_lr/{root_dir}/test_result_{transfer_flag}/'

    my_logger = MyLog().logger

    # 训练集的X和Y
    train_x, train_y, test_x, test_y = get_train_test_data()

    final_idx = test_x.shape[0]
    end_idx = final_idx if end_idx > final_idx else end_idx
    len_split = int(train_x.shape[0] * select_ratio)  # the number of selected train data

    # 全局迁移策略 需要用到初始的csv
    init_weight_file_name = os.path.join(MODEL_SAVE_PATH, f"0006_{pre_hour}h_global_lr_liblinear_{global_lr_iter}.csv")
    global_init_normalize_weight = pd.read_csv(init_weight_file_name).squeeze().tolist()

    # 读取迭代了k次的相似性度量csv文件
    psm_weight = get_feature_weight_list(metric_iter=learned_metric_iteration)

    # 显示参数信息
    my_logger.warning(
        f"[iter params] - global_lr:{global_lr_iter}, local_lr:{local_lr_iter}, learned_iter:{learned_metric_iteration}, pool_nums:{pool_nums}, start_idx:{start_idx}, end_idx:{end_idx}, transfer_flag:{transfer_flag}")

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
        thread = pool.submit(personalized_modeling, pre_data_select, test_idx, x_test_select)
        thread_list.append(thread)

    wait(thread_list, return_when=ALL_COMPLETED)
    collect()

    run_time = round(time.time() - start_time, 2)
    my_logger.warning(f"build all model need time: {run_time} s")

    # ----- save result -----
    try:
        test_result_csv = os.path.join(TEST_RESULT_PATH, f'0009_{learned_metric_iteration}_{start_idx}_{end_idx}_prob_{transfer_flag}.csv')
        test_result.to_csv(test_result_csv, index=False)
        my_logger.warning(f"save {test_result_csv} success!")
    except Exception as err:
        my_logger.error(err)
        raise err

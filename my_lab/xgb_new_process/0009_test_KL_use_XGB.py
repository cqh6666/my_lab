#encoding=gbk
"""
build personal xgb models for test data with learned KL metric
"""
import os
import pickle
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
import threading
import pandas as pd
import xgboost as xgb
from gc import collect
import warnings
import sys
import time
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
        'nthread': 1,
        'verbosity': 0,
        'eval_metric': 'logloss',
        'seed': 998,
        'tree_method': 'hist'
    }
    num_boost_round = xgb_boost_num
    return params, num_boost_round


def personalized_modeling(pre_data, idx, x_test):
    """build personal model for target sample from test datasets"""

    # personalized_modeling_start_time = time.time()
    similar_rank = pd.DataFrame()

    similar_rank['data_id'] = train_x.index.tolist()
    similar_rank['distance'] = (abs((train_x - pre_data) * feature_weight)).sum(axis=1)

    similar_rank.sort_values('distance', inplace=True)
    similar_rank.reset_index(drop=True, inplace=True)
    select_id = similar_rank.iloc[:len_split, 0].values

    select_train_x = train_x.iloc[select_id, :]
    select_train_y = train_y.iloc[select_id]
    fit_train = select_train_x
    fit_test = x_test

    sample_ki = similar_rank.iloc[:len_split, 1].tolist()
    sample_ki = [(sample_ki[0] + m_sample_weight) / (val + m_sample_weight) for val in sample_ki]

    d_train_local = xgb.DMatrix(fit_train, label=select_train_y, weight=sample_ki)
    params, num_boost_round = get_local_xgb_para()

    # use transform
    xgb_local = xgb.train(params=params,
                          dtrain=d_train_local,
                          num_boost_round=num_boost_round,
                          xgb_model=xgb_model,
                          verbose_eval=False)

    d_test_local = xgb.DMatrix(fit_test)
    prob = xgb_local.predict(d_test_local)

    global_lock.acquire()
    test_result.loc[idx, 'prob'] = prob
    # p_weight.loc[idx, :] = xgb_local.get_score(importance_type='weight')
    global_lock.release()

    # run_time = round(time.time() - personalized_modeling_start_time, 2)
    # my_logger.info(f"idx:{idx} | build time:{run_time}s")


def get_train_test_data(key_component):
    train_x = pd.read_feather(
        os.path.join(DATA_SOURCE_PATH, f"all_x_train_{key_component}.feather"))
    train_y = pd.read_feather(
        os.path.join(DATA_SOURCE_PATH, f"all_y_train_{key_component}.feather"))['Label']
    test_x = pd.read_feather(
        os.path.join(DATA_SOURCE_PATH, f"all_x_test_{key_component}.feather"))
    test_y = pd.read_feather(
        os.path.join(DATA_SOURCE_PATH, f"all_y_test_{key_component}.feather"))['Label']

    return train_x, train_y, test_x, test_y


def get_feature_weight_list():
    # 读取相似性度量
    if learned_metric_iteration == 0:
        # 初始权重csv以全局模型迭代100次的模型的特征重要性,赢在起跑线上。
        file_name = '0007_24h_global_xgb_feature_weight_boost500.csv'
        normalize_weight = pd.read_csv(os.path.join(XGB_MODEL_PATH, file_name))
    else:
        file_name = f'0008_{pre_hour}h_{learned_metric_iteration}_psm_boost{xgb_boost_num}{transfer_flag}.csv'
        normalize_weight = pd.read_csv(os.path.join(PSM_SAVE_PATH, file_name))

    f_weight = pd.read_csv(normalize_weight)
    f_weight = f_weight.squeeze().tolist()
    return f_weight


def params_logger_info_show():
    my_logger.warning(
        f"[xgb  params] - xgb_thread_num:{xgb_thread_num},  xgb_boost_num:{xgb_boost_num}, glo_tl_boost_num:{glo_tl_boost_num}")
    my_logger.warning(
        f"[iter params] - transfer_flag:{transfer_flag}, learned_iter:{learned_metric_iteration}, pool_nums:{pool_nums}, start_idx:{start_idx}, end_idx:{end_idx}")


if __name__ == '__main__':

    my_logger = MyLog().logger

    is_transfer = int(sys.argv[1])
    learned_metric_iteration = str(sys.argv[2])
    start_idx = int(sys.argv[3])
    end_idx = int(sys.argv[4])

    pre_hour = 24
    xgb_thread_num = 1
    select_ratio = 0.1
    m_sample_weight = 0.01
    pool_nums = 20
    xgb_boost_num = 50
    glo_tl_boost_num = 500

    transfer_flag = "transfer" if is_transfer == 1 else "no_transfer"
    # ----- work space -----
    root_dir = f"{pre_hour}h"
    DATA_SOURCE_PATH = f"/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/data/{root_dir}/"  # 训练集的X和Y
    XGB_MODEL_PATH = f'/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/result/psm_with_xgb/{root_dir}/global_model/'
    PSM_SAVE_PATH = f'/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/result/psm_with_xgb/{root_dir}/psm_{transfer_flag}/'
    TEST_RESULT_PATH = f'/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/result/psm_with_xgb/{root_dir}/test_result_{transfer_flag}'

    file_flag = f"{pre_hour}_df_rm1_miss2_norm1"
    # 训练集的X和Y
    train_x, train_y, test_x, test_y = get_train_test_data(file_flag)

    final_idx = test_x.shape[0]
    end_idx = final_idx if end_idx > final_idx else end_idx  # 不得大过最大值
    len_split = int(train_x.shape[0] * select_ratio)  # the number of selected train data

    # 迁移模型
    if is_transfer == 1:
        xgb_model_file = os.path.join(XGB_MODEL_PATH, f"0007_{pre_hour}h_global_xgb_boost{glo_tl_boost_num}.pkl")
        xgb_model = pickle.load(open(xgb_model_file, "rb"))
    else:
        xgb_model = None

    # 读取迭代了k次的特征权重csv文件
    feature_weight = get_feature_weight_list()

    # 显示参数信息
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
        thread = pool.submit(personalized_modeling, pre_data_select, test_idx, x_test_select)
        thread_list.append(thread)

    # wait for all threads completing
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

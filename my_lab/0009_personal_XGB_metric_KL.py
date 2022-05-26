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
my_logger = MyLog().logger


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
        'nthread': 2,
        'verbosity': 0,
        'eval_metric': 'logloss',
        'seed': 998,
        'tree_method': 'hist'
    }
    num_boost_round = xgb_boost_num
    return params, num_boost_round


def personalized_modeling(pre_data, idx, x_test):
    """build personal model for target sample from test datasets"""

    personalized_modeling_start_time = time.time()
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

    d_train_local = xgb.DMatrix(fit_train, label=select_train_y, weight=sample_ki)
    params, num_boost_round = get_local_xgb_para()

    # use transform
    xgb_local = xgb.train(params=params,
                          dtrain=d_train_local,
                          num_boost_round=num_boost_round,
                          xgb_model=xgb_model,
                          verbose_eval=False)

    d_test_local = xgb.DMatrix(fit_test)
    proba = xgb_local.predict(d_test_local)

    global_lock.acquire()
    test_result.loc[idx, 'proba'] = proba
    # p_weight.loc[idx, :] = xgb_local.get_score(importance_type='weight')
    global_lock.release()

    # run_time = round(time.time() - personalized_modeling_start_time, 2)
    # my_logger.info(f"idx:{idx} | build time:{run_time}s")


def get_train_test_data():
    train_x = pd.read_feather(
        os.path.join(DATA_SOURCE_PATH, "all_x_train_24h_norm_dataframe_999_miss_medpx_max2dist.feather"))
    train_y = pd.read_feather(
        os.path.join(DATA_SOURCE_PATH, "all_y_train_24h_norm_dataframe_999_miss_medpx_max2dist.feather"))['Label']
    test_x = pd.read_feather(
        os.path.join(DATA_SOURCE_PATH, "all_x_test_24h_norm_dataframe_999_miss_medpx_max2dist.feather"))
    test_y = pd.read_feather(
        os.path.join(DATA_SOURCE_PATH, "all_y_test_24h_norm_dataframe_999_miss_medpx_max2dist.feather"))['Label']

    return train_x, train_y, test_x, test_y


def get_feature_weight_list(metric_iter, tl_boost_num, local_boost_num):
    # ��ȡ�����Զ���
    if learned_metric_iteration == "0":
        feature_importance_file = os.path.join(XGB_MODEL_PATH, '0006_xgb_global_feature_weight_boost100.csv')
    else:
        weight_file_name = f'0008_24h_{metric_iter}_feature_weight_gtlboost{tl_boost_num}_localboost{local_boost_num}.csv'
        feature_importance_file = os.path.join(PSM_SAVE_PATH, weight_file_name)

    f_weight = pd.read_csv(feature_importance_file)
    f_weight = f_weight.squeeze().tolist()
    return f_weight


def params_logger_info_show():
    my_logger.warning(
        f"[xgb  params] - xgb_thread_num:{xgb_thread_num},  xgb_boost_num:{xgb_boost_num}, glo_tl_boost_num:{glo_tl_boost_num}")
    my_logger.warning(
        f"[iter params] - learned_iter:{learned_metric_iteration}, pool_nums:{pool_nums}, start_idx:{start_idx}, end_idx:{end_idx}, transfer_flag:transfer")


if __name__ == '__main__':

    start_idx = int(sys.argv[1])
    end_idx = int(sys.argv[2])
    learned_metric_iteration = str(sys.argv[3])

    xgb_thread_num = 1
    select_ratio = 0.1
    m_sample_weight = 0.01
    pool_nums = 20
    xgb_boost_num = 50
    glo_tl_boost_num = 20

    # ----- work space -----
    DATA_SOURCE_PATH = f"/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/data/24h/"  # ѵ������X��Y
    XGB_MODEL_PATH = '/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/result/personal_model_with_xgb/24h_xgb_model/'

    PSM_SAVE_PATH = f'/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/result/personal_model_with_xgb/24h_xgb_model/24h_transfer_psm/'
    TEST_RESULT_PATH = f'/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/result/personal_model_with_xgb/24h_xgb_model/24h_test_result_transfer/'

    # ѵ������X��Y
    train_x, train_y, test_x, test_y = get_train_test_data()

    final_idx = test_x.shape[0]
    end_idx = final_idx if end_idx > final_idx else end_idx  # ���ô�����ֵ
    len_split = int(train_x.shape[0] * select_ratio)  # the number of selected train data

    # Ǩ��ģ��
    xgb_model_file = os.path.join(XGB_MODEL_PATH, f"0006_xgb_global_24h_all_999_norm_miss_boost{glo_tl_boost_num}.pkl")
    xgb_model = pickle.load(open(xgb_model_file, "rb"))

    # ��ȡ������k�ε�����Ȩ��csv�ļ�
    feature_weight = get_feature_weight_list(metric_iter=learned_metric_iteration, tl_boost_num=glo_tl_boost_num, local_boost_num=xgb_boost_num)

    # ��ʾ������Ϣ
    params_logger_info_show()

    # init test result
    test_result = pd.DataFrame(columns=['real', 'proba'])
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
        test_result_csv = os.path.join(TEST_RESULT_PATH, f'0009_{learned_metric_iteration}_{start_idx}_{end_idx}_proba_transfer.csv')
        test_result.to_csv(test_result_csv, index=False)
        my_logger.warning(f"save {test_result_csv} success!")
    except Exception as err:
        my_logger.error(err)
        raise err

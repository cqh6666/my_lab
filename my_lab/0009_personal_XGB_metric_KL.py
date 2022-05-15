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


def get_global_xgb_para():
    """global xgb para"""
    params = {
        'booster': 'gbtree',
        'max_depth': 8,
        'min_child_weight': 7,
        'eta': 0.15,
        'objective': 'binary:logistic',
        'nthread': xgb_thread_num,
        'verbosity': 0,
        'eval_metric': 'logloss',
        'subsample': 0.5,
        'colsample_bytree': 0.5,
        'tree_method': 'hist',
        'seed': 1001,
    }
    num_boost_round = 50
    return params, num_boost_round


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
    num_boost_round = 50
    return params, num_boost_round


def get_global_xgb():
    """train a global xgb model"""
    params, num_boost_round = get_global_xgb_para()
    d_train_global = xgb.DMatrix(data=train_x, label=train_y)
    model = xgb.train(params=params,
                      dtrain=d_train_global,
                      num_boost_round=num_boost_round,
                      verbose_eval=False)
    return model


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

    run_time = round(time.time() - personalized_modeling_start_time, 2)
    my_logger.info(f"idx:{idx} | build time:{run_time}s")


if __name__ == '__main__':

    start_idx = int(sys.argv[1])
    end_idx = int(sys.argv[2])
    learned_metric_iteration = str(sys.argv[3])

    # ----- work space -----
    ROOT_PATH = '/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/data/24h/'
    WEIGHT_CSV_PATH = '/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/result/personal_model_with_xgb/'

    # 训练集的X和Y
    train_data_x_file = os.path.join(ROOT_PATH, '24h_all_999_normalize_train_x_data.feather')
    train_data_y_file = os.path.join(ROOT_PATH, '24h_all_999_normalize_train_y_data.feather')
    test_data_x_file = os.path.join(ROOT_PATH, '24h_all_999_normalize_test_x_data.feather')
    test_data_y_file = os.path.join(ROOT_PATH, '24h_all_999_normalize_test_y_data.feather')

    # 0008_24h_{iteration_idx}_feature_weight_initboost91_localboost{xgb_boost_num}_mt.csv 读取迭代了k次的特征权重csv文件
    feature_importance_file = os.path.join(WEIGHT_CSV_PATH, f'0008_24h_{learned_metric_iteration}_feature_weight_initboost91_localboost50_mt.csv')
    # 迁移模型
    xgb_model_file = '/panfs/pfs.local/work/liu/xzhang_sta/tangxizhuo/pg/global_model/0006_24h_xgb_glo4_div1_snap1_rm1_miss2_norm1.pkl'

    train_x = pd.read_feather(train_data_x_file)
    train_y = pd.read_feather(train_data_y_file)['Label']
    test_x = pd.read_feather(test_data_x_file)
    test_y = pd.read_feather(test_data_y_file)['Label']
    feature_weight = pd.read_csv(feature_importance_file)
    feature_weight = feature_weight.iloc[:, 0].tolist()

    # personal para setting
    xgb_thread_num = 2
    select_ratio = 0.1
    m_sample_weight = 0.01
    n_thread = 20

    final_idx = test_x.shape[0]
    # start_idx = 0
    # end_idx = 20
    end_idx = final_idx if end_idx > final_idx else end_idx
    my_logger.warning(f"the idx range is: [{start_idx},{end_idx}]")

    # the number of selected train data
    len_split = int(train_x.shape[0] * select_ratio)

    # init test result
    test_result = pd.DataFrame(columns=['real', 'proba'])
    test_result['real'] = test_y.iloc[start_idx:end_idx]

    # init p_weight to save weight importance for each personalized model
    # p_weight = pd.DataFrame(index=test_result.index.tolist(), columns=test_x.columns.tolist())

    # get thread lock
    global_lock = threading.Lock()

    # get global xgb model to transfer learning
    xgb_model = pickle.load(open(xgb_model_file, "rb"))

    start_time = time.time()
    # init thread list
    thread_list = []
    pool = ThreadPoolExecutor(max_workers=n_thread)
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
    my_logger.warning(f"build all model need time: {run_time}s")

    # ----- save result -----
    try:
        test_result_csv = os.path.join(WEIGHT_CSV_PATH, f'24h_transfer_xgb_test_result/0009_{learned_metric_iteration}_proba_tran_{start_idx}_{end_idx}.csv')
        test_result.to_csv(test_result_csv, index=True)
        my_logger.warning(f"save {test_result_csv} success!")
    except Exception as err:
        my_logger.error(err)
        raise err

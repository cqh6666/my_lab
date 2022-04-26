# encoding=gbk
"""
learn KL metric through building xgb personal model
"""
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
import numpy as np
import pandas as pd
from random import shuffle
import xgboost as xgb
from gc import collect
import warnings
import threading
from my_logger import MyLog
import time
import os

warnings.filterwarnings('ignore')


def get_global_xgb_para():
    """global xgb para"""
    params = {
        'booster': 'gbtree',
        'max_depth': 11,
        'min_child_weight': 7,
        'eta': 0.05,
        'objective': 'binary:logistic',
        'nthread': 20,
        'verbosity': 0,
        'eval_metric': 'logloss',
        'subsample': 1,
        'colsample_bytree': 0.7,
        'seed': 1001,
    }
    num_boost_round = 300
    return params, num_boost_round


def get_local_xgb_para():
    """personal xgb para"""
    params = {
        'booster': 'gbtree',
        'max_depth': 11,
        'min_child_weight': 7,
        'eta': 0.05,
        'objective': 'binary:logistic',
        'nthread': 1,
        'verbosity': 0,
        'eval_metric': 'logloss',
        'subsample': 1,
        'colsample_bytree': 0.7,
        'tree_method': 'hist',
        'seed': 1001,
    }
    num_boost_round = 1
    return params, num_boost_round


# def get_xgb_global_weight(x_train, y_train):
#     # ----- init feature weight according to global xgb model -----
#     d_train_all = xgb.DMatrix(x_train, label=y_train)
#     params, num_boost_round = get_global_xgb_para()
#     xgb_all = xgb.train(params=params,
#                         dtrain=d_train_all,
#                         num_boost_round=num_boost_round,
#                         verbose_eval=False)
#     weight_importance = xgb_all.get_score(importance_type='weight')
#     return weight_importance
#
#
# def get_normalize_xgb_weight(weight):
#     """return a pd.Series"""
#     # all_feature: pd.Series, save all feature names
#     all_feature = pd.read_csv('/panfs/pfs.local/work/liu/xzhang_sta/tangxizhuo/data/24h_999_remained_feature.csv', header=None)[0]
#     result = pd.Series(index=all_feature)
#     # drop ID and Label
#     result.drop(['ID', 'Label'], axis=0, inplace=True)
#     # transform dict to pd.Series
#     weight = pd.Series(weight)
#     # len(result) usually > len(weight), extra values will be nan
#     result.loc[:] = weight
#     # normalize feature weight, sum(feature weight) = 1
#     result = result / result.sum()
#     result.fillna(0, inplace=True)
#     return result


# def get_init_normalize_weight(x_train=None, y_train=None, file=None):
#     """return a pd.Series"""
#     if file is not None:
#         return pd.read_pickle(file)
#     return get_normalize_xgb_weight(get_xgb_global_weight(x_train, y_train))


def learn_similarity_measure(pre_data, true, I_idx, X_test):
    lsm_start_time = time.time()

    similar_rank = pd.DataFrame()

    similar_rank['data_id'] = train_rank_x.index.tolist()
    similar_rank['Distance'] = (abs((train_rank_x - pre_data) * normalize_weight)).sum(axis=1)

    similar_rank.sort_values('Distance', inplace=True)
    similar_rank.reset_index(drop=True, inplace=True)
    select_id = similar_rank.iloc[:len_split, 0].values

    x_train = train_rank_x.iloc[select_id, :]
    fit_train = x_train
    y_train = train_rank_y.iloc[select_id]

    fit_test = X_test

    sample_ki = similar_rank.iloc[:len_split, 1].tolist()
    sample_ki = [(sample_ki[0] + m_sample_weight) / (val + m_sample_weight) for val in sample_ki]
    d_train_local = xgb.DMatrix(fit_train, label=y_train, weight=sample_ki)
    params, num_boost_round = get_local_xgb_para()

    xgb_local = xgb.train(params=params,
                          dtrain=d_train_local,
                          num_boost_round=num_boost_round,
                          verbose_eval=False)

    d_test_local = xgb.DMatrix(fit_test)
    proba = xgb_local.predict(d_test_local)

    x_train = x_train - pre_data
    x_train = abs(x_train)
    mean_r = np.mean(x_train)
    y = abs(true - proba)

    global iteration_data
    global y_iteration

    global_lock.acquire()
    iteration_data.loc[I_idx, :] = mean_r
    y_iteration[I_idx] = y
    global_lock.release()

    run_time = round(time.time() - lsm_start_time, 2)
    current_thread = threading.current_thread().getName()
    my_logger.info(
        f"pid:{os.getpid()} | thread:{current_thread} | time:{run_time} s")


# ----- work space -----
my_logger = MyLog().logger

# 训练集的X和Y
train_data_x_file = '/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/data/24h/24h_all_999_normalize_train_x_data.feather'
train_data_y_file = '/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/data/24h/24h_all_999_normalize_train_y_data.feather'
# SAVE_PATH = '/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/result/personal_model_with_xgb/'
# file_name = '0006_xgb_global_feature_weight_importance_boost91_v0.csv'
#
# # ----- get data and init weight ----
train_x = pd.read_feather(train_data_x_file)
train_y = pd.read_feather(train_data_y_file)['Label']
# normalize_weight = pd.read_csv(os.path.join(SAVE_PATH, file_name)).squeeze('columns')

# get data and init weight
# train_x = pd.read_feather(
#     f'/panfs/pfs.local/work/liu/xzhang_sta/tangxizhuo/data/24/24h_train_x_{key_component_name}.feather')
# train_y = pd.read_feather(
#     f'/panfs/pfs.local/work/liu/xzhang_sta/tangxizhuo/data/24/24h_train_y_{key_component_name}.feather')['Label']
key_component_name = 'div1_snap1_rm1_miss2_norm1'
normalize_weight = pd.read_csv(
    f'/panfs/pfs.local/work/liu/xzhang_sta/tangxizhuo/pg/0008_24h_xgb_weight_glo2_{key_component_name}.csv',
    index_col=0).squeeze('columns')

my_logger.info(f"train_x:{train_x.shape} | train_y:{train_y.shape} | normalize_weight:{normalize_weight.shape}")

# ----- similarity learning para -----
# init lock
global_lock = threading.Lock()
# current iteration
cur_iteration = 1
# every {step} iterations, updates normalize_weight in similarity learning
step = 5
# each iteration uses 1000 target samples to build 1000 personal models
n_personal_model_each_iteration = 1000

l_rate = 0.001
select_rate = 0.1
regularization_c = 0.05
m_sample_weight = 0.01

# ----- iteration -----
# one iteration includes 1000 personal models
for iteration_idx in range(cur_iteration, cur_iteration + step):
    last_idx = list(range(train_x.shape[0]))
    shuffle(last_idx)

    last_x = train_x
    last_x = last_x.loc[last_idx, :]
    last_x.reset_index(drop=True, inplace=True)
    last_y = train_y.loc[last_idx]
    last_y.reset_index(drop=True, inplace=True)

    iteration_data = pd.DataFrame(index=range(n_personal_model_each_iteration), columns=train_x.columns)
    y_iteration = pd.Series(index=range(n_personal_model_each_iteration))

    select_x = last_x.loc[:n_personal_model_each_iteration - 1, :]
    select_x.reset_index(drop=True, inplace=True)
    select_y = last_y.loc[:n_personal_model_each_iteration - 1]
    select_y.reset_index(drop=True, inplace=True)

    train_rank_x = last_x.loc[n_personal_model_each_iteration - 1:, :].copy()
    train_rank_x.reset_index(drop=True, inplace=True)
    len_split = int(train_rank_x.shape[0] * select_rate)

    train_rank_y = last_y.loc[n_personal_model_each_iteration - 1:].copy()
    train_rank_y.reset_index(drop=True, inplace=True)

    idx_now = 0

    pg_start_time = time.time()
    my_logger.info(f"iter idx: {iteration_idx} | start building {n_personal_model_each_iteration} models ... ")

    # build n_personal_model_each_iteration personal xgb
    pool = ThreadPoolExecutor(max_workers=20)
    thread_list = []
    for s_idx in range(n_personal_model_each_iteration):
        pre_data_select = select_x.loc[s_idx, :]
        true_select = select_y.loc[s_idx]
        x_test_select = select_x.loc[[s_idx], :]

        thread = pool.submit(learn_similarity_measure, pre_data_select, true_select, idx_now, x_test_select)
        thread_list.append(thread)
        idx_now += 1
        collect()
    # wait for all threads completing
    wait(thread_list, return_when=ALL_COMPLETED)

    run_time = round(time.time() - pg_start_time, 2)
    my_logger.info(
        f"iter idx:{iteration_idx} | build {n_personal_model_each_iteration} models need: {run_time}s")

    # ----- update normalize weight -----
    new_similar = iteration_data * normalize_weight
    y_pred = new_similar.sum(axis=1)

    new_ki = []
    risk_gap = [real - pred for real, pred in zip(list(y_iteration), list(y_pred))]
    for idx, value in enumerate(normalize_weight):
        features_x = list(iteration_data.iloc[:, idx])
        plus_list = [a * b for a, b in zip(risk_gap, features_x)]
        new_value = value + l_rate * (sum(plus_list) - regularization_c * value)
        new_ki.append(new_value)

    new_ki = list(map(lambda x: x if x > 0 else 0, new_ki))
    normalize_weight = new_ki.copy()

    table = pd.DataFrame({'Ma_update_{}'.format(iteration_idx): normalize_weight})
    # file name para: {select ratio}_{m sample weight}_{regularization}-{KL ID}_{iteration idx}
    table.to_csv(
        '/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/result/0008_01_001_005-1_{}.csv'.format(iteration_idx),
        index=False)

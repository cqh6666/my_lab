# encoding=gbk
import time
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
import numpy as np
import pandas as pd
from random import shuffle
import xgboost as xgb
from gc import collect
import warnings
import os
from my_logger import MyLog

warnings.filterwarnings('ignore')

my_logger = MyLog().logger

# 根路径
ROOT_PATH = '/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/data/24h/'
# 经过99.9筛选后的特征集合
remain_feaure_file = os.path.join(ROOT_PATH, '24h_999_remained_feature.csv')
# 训练集的X和Y
train_data_x_file = os.path.join(ROOT_PATH, '24h_all_999_normalize_train_x_data.feather')
train_data_y_file = os.path.join(ROOT_PATH, '24h_all_999_normalize_train_y_data.feather')

# 保存路径
SAVE_PATH = '/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/result/personal_model_with_xgb/'


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
    num_boost_round = 10
    return params, num_boost_round


def get_local_xgb_para():
    """personal xgb para"""
    params = {
        'booster': 'gbtree',
        'max_depth': 8,
        'min_child_weight': 10,
        'subsample': 0.5,
        'colsample_bytree': 0.5,
        'eta': 0.15,
        'objective': 'binary:logistic',
        'nthread': 20,
        'verbosity': 0,
        'eval_metric': 'logloss',
        'seed': 1001,
        'tree_method': 'hist'
    }
    num_boost_round = 1
    return params, num_boost_round


def get_xgb_global_weight(x_train, y_train):
    # ----- init feature weight according to global xgb model -----
    d_train_all = xgb.DMatrix(x_train, label=y_train)
    params, num_boost_round = get_global_xgb_para()
    xgb_all = xgb.train(params=params,
                        dtrain=d_train_all,
                        num_boost_round=num_boost_round,
                        verbose_eval=False)
    weight_importance = xgb_all.get_score(importance_type='weight')
    return weight_importance


def get_normalize_xgb_weight(weight):
    """return a pd.Series"""
    # all_feature: pd.Series, save all feature names
    all_feature = pd.read_csv(remain_feaure_file, header=None)[0]
    result = pd.Series(index=all_feature)
    # drop ID and Label
    result.drop(['ID', 'Label'], axis=0, inplace=True)
    # transform dict to pd.Series
    weight = pd.Series(weight)
    # len(result) usually > len(weight), extra values will be nan
    result.loc[:] = weight
    # normalize feature weight, sum(feature weight) = 1
    result = result / result.sum()
    result.fillna(0, inplace=True)
    return result


def get_init_normalize_weight(x_train=None, y_train=None, file=None):
    """return a pd.Series"""
    if file is not None:
        return pd.read_csv(file, header=None)
    return get_normalize_xgb_weight(get_xgb_global_weight(x_train, y_train))


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
    xgb_global = xgb.train(params=params,
                           dtrain=d_train_local,
                           num_boost_round=num_boost_round,
                           verbose_eval=False)
    d_test_local = xgb.DMatrix(fit_test)
    proba = xgb_global.predict(d_test_local)

    x_train = x_train - pre_data
    x_train = abs(x_train)
    mean_r = np.mean(x_train)
    y = abs(true - proba)

    global iteration_data
    global y_iteration

    # global_lock.acquire()

    iteration_data.loc[I_idx, :] = mean_r
    y_iteration[I_idx] = y
    # global_lock.release()

    my_logger.info(f"a personal model build time : {time.time() - lsm_start_time}")


# ----- work space -----

# get data and init weight
train_x = pd.read_feather(train_data_x_file)
train_y = pd.read_feather(train_data_y_file)['Label']

# ----- similarity learning para -----
# last and current iteration
last_iteration = 0
cur_iteration = last_iteration + 1
# every {step} iterations, updates normalize_weight in similarity learning
step = 5
# each iteration uses 1000 target samples to build 1000 personal models
n_personal_model_each_iteration = 1000

l_rate = 0.00001
select_rate = 0.1
regularization_c = 0.05
m_sample_weight = 0.01

# ----- init weight -----
if last_iteration == 0:
    file_name = '0006_xgb_global_feature_weight_importance_boost91_v0.csv'
    normalize_weight = get_init_normalize_weight(file=os.path.join(SAVE_PATH, file_name))
else:
    file_name = '0008_24h_{}_feature_importance_v0.csv'.format(last_iteration)
    normalize_weight = pd.read_csv(os.path.join(SAVE_PATH, file_name))
    normalize_weight = normalize_weight['Ma_update_{}'.format(last_iteration)].tolist()

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

    # build n_personal_model_each_iteration personal xgb
    pool = ThreadPoolExecutor(max_workers=20)
    thread_list = []
    # 1000 models
    start_time = time.time()
    for s_idx in range(n_personal_model_each_iteration):
        pre_data_select = select_x.loc[s_idx, :]
        true_select = select_y.loc[s_idx]
        x_test_select = select_x.loc[[s_idx], :]

        thread = pool.submit(learn_similarity_measure, pre_data_select, true_select, idx_now, x_test_select)
        thread_list.append(thread)
        idx_now += 1
        # collect()

    wait(thread_list, return_when=ALL_COMPLETED)
    collect()

    pg_end_time = time.time()
    my_logger.info(f"{iteration_idx} build 1000 models need: {pg_end_time}")

    # ----- update normalize weight -----
    # 1000 * columns     columns
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

    if iteration_idx % step == 0:
        table = pd.DataFrame({'Ma_update_{}'.format(iteration_idx): normalize_weight})
        # file name para: {select ratio}_{m sample weight}_{regularization}-{KL ID}_{iteration idx}
        file_name = '0008_24h_{}_feature_weight_initboost91_localboost1.csv'.format(iteration_idx)
        table.to_csv(os.path.join(SAVE_PATH, file_name), index=False)
        my_logger.info(f"save {file_name} success!")


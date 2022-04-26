# encoding=gbk
"""
������ܳ���
"""
import time
import numpy as np
import pandas as pd
from random import shuffle
import xgboost as xgb
import warnings
import os
from my_logger import MyLog
from multiprocessing import cpu_count, Pool, Manager
import psutil
import sys
from gc import collect
warnings.filterwarnings('ignore')

# ��·��
ROOT_PATH = '/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/data/24h/'
# ����99.9ɸѡ�����������
remain_feaure_file = os.path.join(ROOT_PATH, '24h_999_remained_feature.csv')
# ѵ������X��Y
train_data_x_file = os.path.join(ROOT_PATH, '24h_all_999_normalize_train_x_data.feather')
train_data_y_file = os.path.join(ROOT_PATH, '24h_all_999_normalize_train_y_data.feather')

# ����·��
SAVE_PATH = '/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/result/personal_model_with_xgb/'


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
        'nthread': 10,
        'verbosity': 0,
        'eval_metric': 'logloss',
        'seed': 998,
        'tree_method': 'hist'
    }
    num_boost_round = xgb_boost_num
    return params, num_boost_round


def learn_similarity_measure(pre_data, true, I_idx, X_test, global_ns, write_lock):
    """
    ѧϰ�����Զ���
    :param pre_data: Ŀ�껼�ߵ�������
    :param true: Ŀ�껼�ߵı�ǩֵ
    :param I_idx: Ŀ�껼�ߵ�����(0-1000)
    :param X_test: dataFrame��ʽ��Ŀ�껼��������
    :param global_ns: ����̹��������ռ�
    :param write_lock: ����̽���д����ʱ�õ�д��
    :return:
    """
    lsm_start_time = time.time()

    similar_rank = pd.DataFrame()

    similar_rank['data_id'] = train_rank_x.index.tolist()
    similar_rank['Distance'] = (abs((train_rank_x - pre_data) * normalize_weight)).sum(axis=1)

    similar_rank.sort_values('Distance', inplace=True)
    similar_rank.reset_index(drop=True, inplace=True)
    # ѡ��������ǰlen_split������ ����numpy��ʽ
    select_id = similar_rank.iloc[:len_split, 0].values

    x_train = train_rank_x.iloc[select_id, :]
    fit_train = x_train
    y_train = train_rank_y.iloc[select_id]
    # the chosen one
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
    predict_prob = xgb_local.predict(d_test_local)

    # len_split���� - 1���� = �Զ���չΪlen_split  ����ÿ�������Ĳ���
    x_train = x_train - pre_data
    x_train = abs(x_train)
    # ��������ľ�ֵ
    mean_r = np.mean(x_train)
    y = abs(true - predict_prob)

    try:
        write_lock.acquire()
        iter_data = global_ns.iteration_data
        iter_data.loc[I_idx, :] = mean_r
        global_ns.iteration_data = iter_data

        iter_y = global_ns.iteration_y
        iter_y[I_idx] = y
        global_ns.iteration_y = iter_y
    except Exception as err:
        raise err
    finally:
        write_lock.release()

    mem_used = round(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024, 2)
    run_time = round(time.time() - lsm_start_time, 2)
    my_logger.info(f"idx:{I_idx} | pid:{os.getpid()} |  mem_used:{mem_used} GB | time:{run_time} s")


if __name__ == '__main__':

    # ----- work space -----
    # �����Զ�����־��
    my_logger = MyLog().logger

    memory_total = round(psutil.virtual_memory().total / 1024 / 1024 / 1024, 2)
    memory_used = round(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024, 2)
    memory_percent = round(psutil.Process(os.getpid()).memory_percent(), 2)
    my_logger.warn(f"cpu_count: {cpu_count()}")
    my_logger.warn(f"mem total: {memory_total} G | mem used: {memory_used} G | percent: {memory_percent} %")

    # ----- get data and init weight ----
    train_x = pd.read_feather(train_data_x_file)
    train_y = pd.read_feather(train_data_y_file)['Label']

    # ----- similarity learning para -----
    # last and current iteration
    # every {step} iterations, updates normalize_weight in similarity learning
    init_iteration = int(sys.argv[1])
    cur_iteration = init_iteration + 1
    step = 5

    l_rate = 0.001
    select_rate = 0.1
    regularization_c = 0.05
    m_sample_weight = 0.01

    xgb_boost_num = 1
    pool_nums = 20
    max_tasks_per_child = None
    n_personal_model_each_iteration = 1000

    # ----- init weight -----
    if init_iteration == 0:
        # file_name = '0006_xgb_global_feature_weight_importance_boost91_v0.csv'
        # dataframe -> series
        # normalize_weight = pd.read_csv(os.path.join(SAVE_PATH, file_name))
        file_name = '0008_24h_xgb_weight_glo2_div1_snap1_rm1_miss2_norm1.csv'
        normalize_weight = pd.read_csv(os.path.join(SAVE_PATH, file_name), index_col=0)
    else:
        file_name = f'0008_24h_{init_iteration}_feature_weight_initboost91_localboost{xgb_boost_num}_mp.csv'
        normalize_weight = pd.read_csv(os.path.join(SAVE_PATH, file_name))

    my_logger.warn(f"normalize_weight shape:{normalize_weight.shape} ")

    last_idx = list(range(train_x.shape[0]))
    shuffle(last_idx)
    last_x = train_x.loc[last_idx, :]
    last_x.reset_index(drop=True, inplace=True)
    last_y = train_y.loc[last_idx]
    last_y.reset_index(drop=True, inplace=True)

    select_x = last_x.loc[:n_personal_model_each_iteration - 1, :]
    select_x.reset_index(drop=True, inplace=True)
    select_y = last_y.loc[:n_personal_model_each_iteration - 1]
    select_y.reset_index(drop=True, inplace=True)

    train_rank_x = last_x.loc[n_personal_model_each_iteration - 1:, :].copy()
    train_rank_x.reset_index(drop=True, inplace=True)
    train_rank_y = last_y.loc[n_personal_model_each_iteration - 1:].copy()
    train_rank_y.reset_index(drop=True, inplace=True)

    len_split = int(train_rank_x.shape[0] * select_rate)

    my_logger.warn(f"start iteration ... ")

    # ----- iteration -----
    # one iteration includes 1000 personal models
    for iteration_idx in range(cur_iteration, cur_iteration + step):

        iteration_data = pd.DataFrame(index=range(n_personal_model_each_iteration), columns=train_x.columns)
        iteration_y = pd.Series(index=range(n_personal_model_each_iteration))

        # [0,1,2,...,n]
        # build n_personal_model_each_iteration personal xgb
        pool = Pool(processes=pool_nums, maxtasksperchild=max_tasks_per_child)

        # global namespace
        global_manager = Manager()
        # д����ʱ��Ҫ�������������©д���
        lock = global_manager.Lock()
        # ���������ռ�,����̻��õ��Ĺ�������
        global_namespace = global_manager.Namespace()
        global_namespace.iteration_data = iteration_data
        global_namespace.iteration_y = iteration_y

        my_logger.warn(f"iter idx: {iteration_idx} | start building {n_personal_model_each_iteration} models ... ")
        # 1000 models
        pg_start_time = time.time()
        for s_idx in range(n_personal_model_each_iteration):
            pre_data_select = select_x.loc[s_idx, :]
            true_select = select_y.loc[s_idx]
            # dataframe��ʽ 1000�������еı�ѡ�����һ��
            x_test_select = select_x.loc[[s_idx], :]

            pool.apply_async(func=learn_similarity_measure,
                             args=(pre_data_select, true_select, s_idx, x_test_select, global_namespace, lock))
        pool.close()
        pool.join()

        run_time = round(time.time() - pg_start_time, 2)
        my_logger.warn(
            f"iter idx:{iteration_idx} | build {n_personal_model_each_iteration} models need: {run_time}s")

        # ----- update normalize weight -----
        # 1000 * columns     columns
        iteration_data = global_namespace.iteration_data
        iteration_y = global_namespace.iteration_y
        new_similar = iteration_data * normalize_weight
        y_pred = new_similar.sum(axis=1)

        update_weight_start_time = time.time()
        new_ki = []
        risk_gap = [real - pred for real, pred in zip(list(iteration_y), list(y_pred))]
        for idx, value in enumerate(normalize_weight.squeeze('columns')):
            features_x = list(iteration_data.iloc[:, idx])
            plus_list = [a * b for a, b in zip(risk_gap, features_x)]
            new_value = value + l_rate * (sum(plus_list) - regularization_c * value)
            new_ki.append(new_value)
        new_ki_map = list(map(lambda x: x if x > 0 else 0, new_ki))
        # list -> dataframe
        normalize_weight = pd.DataFrame(
            {'Ma_update_{}'.format(iteration_idx): new_ki_map})
        update_weight_run_time = round(time.time() - update_weight_start_time, 2)
        my_logger.info(
            f"iter idx:{iteration_idx} | update weight need: {update_weight_run_time}s")

        try:
            if iteration_idx % step == 0:
                file_name = f'0008_24h_{iteration_idx}_feature_weight_initboost91_localboost{xgb_boost_num}_mp.csv'
                normalize_weight.to_csv(os.path.join(SAVE_PATH, file_name), index=False)
                my_logger.info(f"iter idx: {iteration_idx} | save {file_name} success!")
        except Exception as err:
            my_logger.error(f"iter idx: {iteration_idx} | save {file_name} error!")
            raise err

        del global_namespace, iteration_data, iteration_y, new_ki
        collect()
        my_logger.info(f"======================= {iteration_idx} rounds done ! ========================")

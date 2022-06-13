# encoding=gbk
"""
多线程跑程序
"""
import threading
import time
import numpy as np
import pandas as pd
from random import shuffle
import xgboost as xgb
import warnings
import os
from my_logger import MyLog
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
import sys
from gc import collect
import pickle
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


def learn_similarity_measure(pre_data, true, I_idx, X_test):
    """
    学习相似性度量
    :param pre_data: 目标患者的特征集
    :param true: 目标患者的标签值
    :param I_idx: 目标患者的索引(0-1000)
    :param X_test: dataFrame格式的目标患者特征集
    :return:
    """
    # lsm_start_time = time.time()

    similar_rank = pd.DataFrame()

    similar_rank['data_id'] = train_rank_x.index.tolist()
    similar_rank['distance'] = (abs((train_rank_x - pre_data) * normalize_weight)).sum(axis=1)

    similar_rank.sort_values('distance', inplace=True)
    similar_rank.reset_index(drop=True, inplace=True)
    # 选出相似性前len_split个样本 返回numpy格式
    select_id = similar_rank.iloc[:len_split, 0].values

    x_train = train_rank_x.iloc[select_id, :]
    y_train = train_rank_y.iloc[select_id]
    fit_train = x_train
    fit_test = X_test

    sample_ki = similar_rank.iloc[:len_split, 1].tolist()
    sample_ki = [(sample_ki[0] + m_sample_weight) / (val + m_sample_weight) for val in sample_ki]
    d_train_local = xgb.DMatrix(fit_train, label=y_train, weight=sample_ki)
    params, num_boost_round = get_local_xgb_para()

    xgb_local = xgb.train(params=params,
                          dtrain=d_train_local,
                          num_boost_round=num_boost_round,
                          verbose_eval=False,
                          xgb_model=xgb_model)

    d_test_local = xgb.DMatrix(fit_test)
    predict_prob = xgb_local.predict(d_test_local)

    # len_split长度 - 1长度 = 自动伸展为len_split  代表每个特征的差异
    x_train = x_train - pre_data
    x_train = abs(x_train)
    # 特征差异的均值
    mean_r = np.mean(x_train)
    y = abs(true - predict_prob)

    global iteration_data
    global iteration_y
    global lock
    try:
        lock.acquire()
        iteration_data.loc[I_idx, :] = mean_r
        iteration_y[I_idx] = y
    except Exception as err:
        my_logger.error(err)
        raise err
    finally:
        lock.release()

    # run_time = round(time.time() - lsm_start_time, 2)
    # current_thread = threading.current_thread().getName()
    # my_logger.info(
    #     f"pid:{os.getpid()} | thread:{current_thread} | time:{run_time} s")


if __name__ == '__main__':

    pre_hour = 24
    root_dir = f"{pre_hour}h_old2"

    is_transfer = int(sys.argv[1])
    init_iteration = int(sys.argv[2])

    transfer_flag = "transfer" if is_transfer == 1 else "no_transfer"
    cur_iteration = init_iteration + 1

    DATA_SOURCE_PATH = f"/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/data/{root_dir}/"
    XGB_MODEL_PATH = f'/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/result/psm_with_xgb/{root_dir}/global_model/'
    PSM_SAVE_PATH = f'/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/result/psm_with_xgb/{root_dir}/psm_{transfer_flag}/'

    # 训练集的X和Y
    key_component = f"{pre_hour}_df_rm1_norm1"
    train_x = pd.read_feather(
        os.path.join(DATA_SOURCE_PATH, f"all_x_train_{key_component}.feather"))
    train_y = pd.read_feather(
        os.path.join(DATA_SOURCE_PATH, f"all_y_train_{key_component}.feather"))['Label']

    # ----- work space -----
    # 引入自定义日志类
    my_logger = MyLog().logger

    # ----- similarity learning para -----
    step = 3
    l_rate = 0.00001
    select_rate = 0.1
    regularization_c = 0.05
    m_sample_weight = 0.01

    xgb_thread_num = 1
    # 做迁移时才会有这个全局模型参数
    glo_tl_boost_num = 500
    xgb_boost_num = 50
    pool_nums = 10
    n_personal_model_each_iteration = 1000

    # 迁移模型
    if is_transfer == 1:
        xgb_model_file = os.path.join(XGB_MODEL_PATH, f"0007_{pre_hour}h_global_xgb_boost{glo_tl_boost_num}.pkl")
        xgb_model = pickle.load(open(xgb_model_file, "rb"))
    else:
        xgb_model = None

    my_logger.warning(
        f"[params] - transfer_flag:{transfer_flag}, init_iter:{init_iteration}, xgb_boost_num:{xgb_boost_num}, pool_nums:{pool_nums}, personal_model:{n_personal_model_each_iteration}")

    # ----- init weight  | dataFrame 格式，有header行，没index索引列-----
    if init_iteration == 0:
        # 初始权重csv以全局模型迭代100次的模型的特征重要性,赢在起跑线上。
        file_name = '0007_24h_global_xgb_feature_weight_boost500.csv'
        normalize_weight = pd.read_csv(os.path.join(XGB_MODEL_PATH, file_name)).squeeze().tolist()
    else:
        file_name = f'0008_{pre_hour}h_{init_iteration}_psm_boost{xgb_boost_num}_{transfer_flag}.csv'
        normalize_weight = pd.read_csv(os.path.join(PSM_SAVE_PATH, file_name)).squeeze().tolist()

    lock = threading.Lock()
    my_logger.warning("start iteration ... ")

    # ----- iteration -----
    # one iteration includes 1000 personal models
    for iteration_idx in range(cur_iteration, cur_iteration + step):
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

        iteration_data = pd.DataFrame(index=range(n_personal_model_each_iteration), columns=train_x.columns)
        iteration_y = pd.Series(index=range(n_personal_model_each_iteration))

        pg_start_time = time.time()

        with ThreadPoolExecutor(max_workers=pool_nums) as executor:
            thread_list = []
            for s_idx in range(n_personal_model_each_iteration):
                pre_data_select = select_x.loc[s_idx, :]
                true_select = select_y.loc[s_idx]
                # dataframe格式 1000个样本中的被选择的那一个
                x_test_select = select_x.loc[[s_idx], :]

                thread = executor.submit(learn_similarity_measure, pre_data_select, true_select, s_idx, x_test_select)
                thread_list.append(thread)
            wait(thread_list, return_when=ALL_COMPLETED)

        run_time = round(time.time() - pg_start_time, 2)
        my_logger.warning(
            f"iter idx:{iteration_idx} | build {n_personal_model_each_iteration} models need: {run_time}s")

        # ----- update normalize weight -----
        # 1000 * columns     columns
        """
        iteration_data 代表目标样本与前N个相似样本的每个特征的差异平均值
        乘上normaliza_weight 代表 代表每个特征有不一样的重要性，重要性高的特征差异就会更大
        all_error 就是计算所有特征差异的权值之和
        """
        new_similar = iteration_data * normalize_weight
        all_error = new_similar.sum(axis=1)

        new_ki = []
        risk_gap = [real - pred for real, pred in zip(list(iteration_y), list(all_error))]
        # 具有单列或单行的数据被Squeeze为一个Series。
        for idx, value in enumerate(normalize_weight):
            features_x = list(iteration_data.iloc[:, idx])
            plus_list = [a * b for a, b in zip(risk_gap, features_x)]
            new_value = value + l_rate * (sum(plus_list) - regularization_c * value)
            new_ki.append(new_value)

        normalize_weight = list(map(lambda x: x if x > 0 else 0, new_ki))
        # list -> dataframe
        normalize_weight_df = pd.DataFrame({'Ma_update_{}'.format(iteration_idx): normalize_weight})

        try:
            # if iteration_idx % step == 0:
            file_name = f'0008_{pre_hour}h_{iteration_idx}_psm_boost{xgb_boost_num}_{transfer_flag}.csv'
            normalize_weight_df.to_csv(os.path.join(PSM_SAVE_PATH, file_name), index=False)
            my_logger.warning(f"iter idx: {iteration_idx} | save {file_name} success!")
        except Exception as err:
            my_logger.error(f"iter idx: {iteration_idx} | save {file_name} error!")
            raise err

        del iteration_data, iteration_y, new_ki
        collect()

        my_logger.warning(f"======================= {iteration_idx} rounds done ! ========================")

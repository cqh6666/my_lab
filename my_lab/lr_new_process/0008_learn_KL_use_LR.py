# encoding=gbk
"""
多线程跑程序
"""
from threading import Lock
import time
import numpy as np
import pandas as pd
from random import shuffle
import warnings
import os

from sklearn.preprocessing import StandardScaler

from my_logger import MyLog
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
import sys
from gc import collect
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings('ignore')


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

    # 10%的数据 个性化建模
    x_train = train_rank_x.iloc[select_id, :]
    y_train = train_rank_y.iloc[select_id]
    if is_transfer == 1:
        init_weight = global_init_normalize_weight
        fit_train = x_train * init_weight
        fit_test = X_test * init_weight
    else:
        fit_train = x_train
        fit_test = X_test

    # 相似样本对应的权重，越相似权重越高
    sample_ki = similar_rank.iloc[:len_split, 1].tolist()
    sample_ki = [(sample_ki[0] + m_sample_weight) / (val + m_sample_weight) for val in sample_ki]

    lr_local = LogisticRegression(solver="liblinear", n_jobs=1, max_iter=local_lr_iter)
    lr_local.fit(fit_train, y_train, sample_ki)

    y_predict = lr_local.predict_proba(fit_test)[:, 1]
    # len_split长度 - 1长度 = 自动伸展为len_split  代表每个特征的差异
    x_train = x_train - pre_data
    x_train = abs(x_train)
    # 特征差异的均值
    mean_r = np.mean(x_train)
    y = abs(true - y_predict)

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
   # my_logger.info(f"[{I_idx}] - train_iter:{lr_local.n_iter_}, cost time: {run_time} s")


if __name__ == '__main__':

    is_transfer = int(sys.argv[1])
    init_iteration = int(sys.argv[2])

    pre_hour = 24
    transfer_flag = "no_transfer" if is_transfer == 0 else "transfer"

    root_dir = f"{pre_hour}h"
    DATA_SOURCE_PATH = f"/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/data/{root_dir}/"  # 训练集的X和Y
    MODEL_SAVE_PATH = f'/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/result/personal_model_with_lr/{root_dir}/global_model/'
    PSM_SAVE_PATH = f'/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/result/personal_model_with_lr/{root_dir}/psm_{transfer_flag}/'

    train_x = pd.read_feather(
        os.path.join(DATA_SOURCE_PATH, f"all_x_train_{pre_hour}h_norm_dataframe_999_miss_medpx_max2dist.feather"))
    train_y = pd.read_feather(
        os.path.join(DATA_SOURCE_PATH, f"all_y_train_{pre_hour}h_norm_dataframe_999_miss_medpx_max2dist.feather"))['Label']

    my_logger = MyLog().logger

    # ----- similarity learning para -----
    # last and current iteration
    # every {step} iterations, updates normalize_weight in similarity learning
    cur_iteration = init_iteration + 1
    step = 10
    l_rate = 0.00001
    select_rate = 0.1
    regularization_c = 0.05
    m_sample_weight = 0.01

    # 不迁移的话设置为20+50
    pool_nums = 25
    n_personal_model_each_iteration = 1000
    global_lr_iter = 400
    local_lr_iter = 50

    my_logger.warning(
        f"[params] - is_transfer:{is_transfer}, init_iter:{init_iteration}, pool_nums:{pool_nums}, n_personal_model:{n_personal_model_each_iteration}, global_lr:{global_lr_iter}, local_lr:{local_lr_iter}")

    # 全局迁移策略 需要用到初始的csv
    init_weight_file_name = os.path.join(MODEL_SAVE_PATH, f"0006_{pre_hour}h_global_lr_{global_lr_iter}.csv")
    global_init_normalize_weight = pd.read_csv(init_weight_file_name).squeeze().tolist()

    # ----- init weight -----
    if init_iteration == 0:
        normalize_weight = global_init_normalize_weight
    else:
        wi_file_name = os.path.join(PSM_SAVE_PATH, f"0008_{pre_hour}h_{init_iteration}_psm_{transfer_flag}.csv")
        normalize_weight = pd.read_csv(wi_file_name).squeeze().tolist()

    lock = Lock()
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
            f"iter idx:{iteration_idx} | build {n_personal_model_each_iteration} models need: {run_time} s")

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
        normalize_weight_df = pd.DataFrame({f'Ma_update_{iteration_idx}': normalize_weight})

        try:
            wi_file_name = os.path.join(PSM_SAVE_PATH, f"0008_{pre_hour}h_{iteration_idx}_psm_{transfer_flag}.csv")
            normalize_weight_df.to_csv(wi_file_name, index=False)
            my_logger.warning(f"iter idx: {iteration_idx} | save {wi_file_name} success!")
        except Exception as err:
            my_logger.error(f"iter idx: {iteration_idx} | save error!")
            raise err

        del iteration_data, iteration_y
        collect()

        my_logger.warning(f"======================= {iteration_idx} rounds done ! ========================")

    print("run done!")
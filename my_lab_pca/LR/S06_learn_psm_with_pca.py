# encoding=gbk
"""
多线程跑LR程序
"""
from threading import Lock
import time
import numpy as np
import pandas as pd
from random import shuffle
import warnings
import os

from sklearn.decomposition import PCA
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
import sys
from gc import collect
from my_logger import MyLog

from utils_api import get_train_test_x_y

from sklearn.linear_model import LogisticRegression
from lr_utils_api import get_init_similar_weight, get_transfer_weight

warnings.filterwarnings('ignore')


def get_similar_rank(target_pre_data_select):
    """
    选择前10%的样本，并且根据相似得到样本权重
    :param target_pre_data_select:
    :return:
    """
    try:
        similar_rank = pd.DataFrame(index=pca_train_data_x.index)
        similar_rank['distance'] = abs(pca_train_data_x - target_pre_data_select.values).sum(axis=1)
        similar_rank.sort_values('distance', inplace=True)
        patient_ids = similar_rank.index[:len_split].values
        sample_ki = similar_rank.iloc[:len_split, 0].values
        sample_ki = [(sample_ki[0] + m_sample_weight) / (val + m_sample_weight) for val in sample_ki]
    except Exception as err:
        print(err)
        sys.exit(1)

    return patient_ids, sample_ki


def fit_train_test_data(select_x_train, pre_data_select_x, is_tra):
    """
    lr专属，迁移需要*全局参数
    :return:
    """
    if is_tra == 1:
        # 全局迁移
        transfer_weight = global_feature_weight
        fit_train_x = select_x_train * transfer_weight
        fit_test_x = pre_data_select_x * transfer_weight
    else:
        fit_train_x = select_x_train
        fit_test_x = pre_data_select_x
    return fit_test_x, fit_train_x


def lr_train(fit_train_x, fit_train_y, pre_data_select_, sample_ki):
    """
    lr 训练 ，得到预测概率
    :return:
    """
    lr_local = LogisticRegression(solver="liblinear", n_jobs=1, max_iter=local_lr_iter)
    lr_local.fit(fit_train_x, fit_train_y, sample_ki)
    predict_prob = lr_local.predict_proba(pre_data_select_)[0][1]
    return predict_prob


def learn_similarity_measure(test_id_, pre_data_select_x, pca_pre_data_select_x, pre_data_y_select_):
    """
    学习相似性度量
    :param test_id_:  目标患者的索引(0-1000)
    :param pre_data_x_select_: 目标患者的特征集 df
    :param pca_pre_data_x_select_: pca降维后的目标患者特征集 df
    :param pre_data_y_select_: 目标患者的标签值
    :return:
    """
    # 找到相似患者ID
    patient_ids, sample_ki = get_similar_rank(pca_pre_data_select_x)

    # 筛选出的相似患者 原始数据
    x_train = train_rank_x.loc[patient_ids, :]
    y_train = train_rank_y.loc[patient_ids]

    # =========================== LR专属处理 ================================#
    fit_train_y = y_train
    fit_test_x, fit_train_x = fit_train_test_data(x_train, pre_data_select_x, is_transfer)
    predict_prob = lr_train(fit_train_x, fit_train_y, fit_test_x, sample_ki)
    # =========================== LR专属处理 ================================#

    # 每个特征的差异的均值 必须是 array 或 list 格式,才能进行扩展加减; 需要.values
    mean_r = np.mean(abs(x_train - pre_data_select_x.values))
    # 特征差异的均值
    y = abs(pre_data_y_select_ - predict_prob)

    global iteration_data
    global iteration_y
    global lock
    try:
        lock.acquire()
        iteration_data.loc[test_id_, :] = mean_r
        iteration_y[test_id_] = y
    except Exception as err:
        print(err)
        sys.exit(1)
    finally:
        lock.release()


def pca_reduction(train_x, test_x, similar_weight, n_comp):
    if n_comp >= train_x.shape[1]:
        n_comp = train_x.shape[1] - 1

    my_logger.warning(f"starting pca by train_data...")
    # pca降维
    pca_model = PCA(n_components=n_comp, random_state=2022)
    # 转换需要 * 相似性度量
    new_train_data_x = pca_model.fit_transform(train_x * similar_weight)
    new_test_data_x = pca_model.transform(test_x * similar_weight)
    # 转成df格式
    pca_train_x = pd.DataFrame(data=new_train_data_x, index=train_x.index)
    pca_test_x = pd.DataFrame(data=new_test_data_x, index=test_x.index)

    my_logger.info(f"n_components: {pca_model.n_components}, svd_solver:{pca_model.svd_solver}.")

    return pca_train_x, pca_test_x


if __name__ == '__main__':

    my_logger = MyLog().logger

    is_transfer = int(sys.argv[1])
    init_iteration = int(sys.argv[2])

    transfer_flag = "no_transfer" if is_transfer == 0 else "transfer"

    PSM_SAVE_PATH = f'./result/S06_temp/psm_{transfer_flag}'
    if not os.path.exists(PSM_SAVE_PATH):
        os.makedirs(PSM_SAVE_PATH)

    train_x, train_y, _, _ = get_train_test_x_y()
    data_columns = train_x.columns

    # ----- similarity learning para -----
    cur_iteration = init_iteration + 1

    l_rate = 0.00001
    select_rate = 0.1
    regularization_c = 0.05
    m_sample_weight = 0.01
    n_personal_model_each_iteration = 1000
    n_components = 100

    # =========================== LR专属 ================================#
    version = 1
    pool_nums = 35
    step = 15
    local_lr_iter = 100
    global_feature_weight = get_transfer_weight(is_transfer)

    my_logger.warning(f"[params] - local:{local_lr_iter}, step:{step}, pool_nums:{pool_nums}, version:{version}")
    # =========================== LR专属 ================================#

    # ================================== save file======================================
    psm_file_name = f"S06_iter{init_iteration}_dim{n_components}_tra{is_transfer}_v{version}.csv"
    # ================================== save file======================================

    my_logger.warning(f"[params] - is_transfer:{is_transfer}, init_iter:{init_iteration}, dim:{n_components}")
    # ----- 相似性度量 -----
    if init_iteration == 0:
        psm_weight = get_init_similar_weight()
    else:
        psm_weight = pd.read_csv(os.path.join(PSM_SAVE_PATH, psm_file_name)).squeeze().tolist()
    my_logger.warning(f"load psm_weight... [iter]:{init_iteration}")

    lock = Lock()
    my_logger.warning("start iteration ... ")

    # ----- iteration -----
    # one iteration includes 1000 personal models
    for iteration_idx in range(cur_iteration, cur_iteration + step):
        last_idx = list(range(train_x.shape[0]))
        # 打乱顺序
        shuffle(last_idx)
        last_x = train_x.loc[last_idx, :]
        last_x.reset_index(drop=True, inplace=True)
        last_y = train_y.loc[last_idx]
        last_y.reset_index(drop=True, inplace=True)

        # 选取1000个患者作为目标患者
        select_x = last_x.loc[:n_personal_model_each_iteration - 1, :].copy()
        select_x.reset_index(drop=True, inplace=True)
        select_y = last_y.loc[:n_personal_model_each_iteration - 1].copy()
        select_y.reset_index(drop=True, inplace=True)

        # 作为训练样本或匹配样本（匹配相似患者的样本集）
        train_rank_x = last_x.loc[n_personal_model_each_iteration - 1:, :].copy()
        train_rank_x.reset_index(drop=True, inplace=True)
        train_rank_y = last_y.loc[n_personal_model_each_iteration - 1:].copy()
        train_rank_y.reset_index(drop=True, inplace=True)

        # pca 降维
        pca_train_data_x, pca_test_data_x = pca_reduction(train_rank_x, select_x, psm_weight, n_components)

        # 选取相似样本集的样本大小 10%
        len_split = int(train_rank_x.shape[0] * select_rate)

        # 保存特征之间的差的平均值 和 保存预测y
        personal_list = range(n_personal_model_each_iteration)  # 也就是test_id_list
        iteration_data = pd.DataFrame(index=personal_list, columns=data_columns)
        iteration_y = pd.Series(index=personal_list)

        pg_start_time = time.time()

        with ThreadPoolExecutor(max_workers=pool_nums) as executor:
            thread_list = []
            for test_id in personal_list:
                # 原始数据的 目标患者
                pre_data_x_select = select_x.loc[[test_id]]
                # pca降维后的 目标患者
                pca_pre_data_x_select = pca_test_data_x.loc[[test_id]]
                pre_data_y_select = select_y.loc[test_id]  # target y label

                thread = executor.submit(learn_similarity_measure, test_id, pre_data_x_select, pca_pre_data_x_select, pre_data_y_select)
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
        # print("============ iteration_data ===============")
        # print(iteration_data.iloc[:10, :10])
        # print("============ iteration_y ===============")
        # print(iteration_y.iloc[:5])
        new_similar = iteration_data * psm_weight
        all_error = new_similar.sum(axis=1)

        new_ki = []
        risk_gap = [real - pred for real, pred in zip(list(iteration_y), list(all_error))]
        # 具有单列或单行的数据被Squeeze为一个Series。
        for idx, value in enumerate(psm_weight):
            features_x = list(iteration_data.iloc[:, idx])
            plus_list = [a * b for a, b in zip(risk_gap, features_x)]
            new_value = value + l_rate * (sum(plus_list) - regularization_c * value)
            new_ki.append(new_value)

        psm_weight = list(map(lambda x: x if x > 0 else 0, new_ki))
        # list -> dataframe
        psm_weight_df = pd.DataFrame({f'psm_update_{iteration_idx}': psm_weight})

        try:
            if iteration_idx % 5 == 0:
                wi_file_name = os.path.join(PSM_SAVE_PATH, f"S06_iter{iteration_idx}_dim{n_components}_tra{is_transfer}_v{version}.csv")
                psm_weight_df.to_csv(wi_file_name, index=False)
                my_logger.warning(f"iter idx: {iteration_idx} | save {wi_file_name} success!")
        except Exception as err:
            my_logger.error(f"iter idx: {iteration_idx} | save error!")

        del iteration_data, iteration_y
        collect()

        my_logger.warning(f"======================= {iteration_idx} rounds done ! ========================")

    print("run done!")

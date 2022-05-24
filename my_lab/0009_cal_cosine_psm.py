# -*- coding: gbk -*-
"""
-------------------------------------------------
   File Name:     0009_cosine_psm
   Description:   ...
   Author:        cqh
   date:          2022/5/19 16:06
-------------------------------------------------
   Change Activity:
                  2022/5/19:
-------------------------------------------------
"""
__author__ = 'cqh'

import sys

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances

import os
import pandas as pd

from my_logger import MyLog


def check_psm_all_file_list(start_idx, max_idx, step_idx):
    """获得psm不同迭代的所有csv文件,确保这些文件都是存在的，并返回所有文件名列表"""

    psm_file_name_list = []
    for cur_idx in range(start_idx, max_idx, step_idx):
        psm_file_name = f"0008_24h_{cur_idx}_{psm_flag}.csv"
        cur_psm_file = os.path.join(PSM_SAVE_PATH, psm_file_name)
        if not os.path.exists(cur_psm_file):
            my_logger.error(f"no find {psm_file_name}, please reset the range idx...")
            return None
        psm_file_name_list.append(psm_file_name)

    return psm_file_name_list


def get_psm_dist(start_idx, max_idx, step_idx, dist_type="cosine"):
    """
    得到psm各自之间的关系矩阵
    :param max_idx:
    :param step_idx:
    :param start_idx: 1
    :param dist_type: cosine or euclidean
    :return:
    """
    # init_weight
    psm_weight = pd.read_csv(os.path.join(INIT_PSM_PATH, init_psm_file)).squeeze()
    all_weight = [psm_weight]

    for cur_idx in range(start_idx, max_idx, step_idx):
        file_flag = f"0008_24h_{cur_idx}_"
        psm_file_name = os.path.join(PSM_SAVE_PATH, f"{file_flag}{psm_flag}.csv")
        cur_weight = pd.read_csv(psm_file_name).squeeze()
        my_logger.warning(f"load {psm_file_name}...")
        all_weight.append(cur_weight)

    if len(all_weight) == 1:
        my_logger.error("no find psm file... stop save result file")
        return

    if dist_type == "cosine":
        save_file_name = os.path.join(PSM_SAVE_PATH, f"psm_{dist_type}_by_iter_{transfer_flag}-{start_idx}_{max_idx}_{step_idx}.csv")
        pd.DataFrame(cosine_similarity(all_weight)).to_csv(save_file_name)
        my_logger.info(f"success save cosine similarity by psm csv! {save_file_name}")
    elif dist_type == "euclidean":
        save_file_name = os.path.join(PSM_SAVE_PATH, f"psm_{dist_type}_by_iter_{transfer_flag}-{start_idx}_{max_idx}_{step_idx}.csv")
        pd.DataFrame(euclidean_distances(all_weight)).to_csv(save_file_name)
        my_logger.info(f"success save euclidean distances by psm csv! {save_file_name}")


def get_feature_psm_std(psm_iter_list, data_source, result_save_file):
    """
    psm * std
    :param psm_iter_list: 相似性度量迭代次数列表
    :param data_source: 数据源
    :param result_save_file: 结果保存文件
    :return:
    """
    data_source_std = data_source.std().tolist()

    feature_result = pd.DataFrame({"std": data_source_std})

    # init weight 0
    psm_weight = pd.read_csv(os.path.join(INIT_PSM_PATH, init_psm_file)).squeeze()
    temp_result = pd.DataFrame({"std": data_source_std, "psm": psm_weight})
    feature_result['0'] = temp_result['std'] * temp_result['psm']

    # iter
    for psm_iter in psm_iter_list:
        psm_file_name = f"0008_24h_{psm_iter}_{psm_flag}.csv"
        psm_weight = pd.read_csv(os.path.join(PSM_SAVE_PATH, psm_file_name)).squeeze()
        temp_result_pd = pd.DataFrame({"std": data_source_std, "psm": psm_weight})
        feature_name = f'iter_{psm_iter}'
        feature_result[feature_name] = temp_result_pd['std'] * temp_result_pd['psm']
        my_logger.warning(f"load {psm_file_name}...")

    print(feature_result.info)
    my_logger.info(f"feature_result shape: {feature_result.shape}")

    feature_result.index = data_source.columns
    save_file = os.path.join(PSM_SAVE_PATH, result_save_file)
    feature_result.to_csv(save_file)
    my_logger.info(f"save success! - {save_file}")


def get_feature_psm_std_run(start_idx, max_idx, step_idx):
    train_x = pd.read_feather(
        os.path.join(DATA_SOURCE_PATH, "all_x_train_24h_norm_dataframe_999_miss_medpx_max2dist.feather"))
    psm_list = [i for i in range(start_idx, max_idx, step_idx)]
    result_save_file = f"psm_std_by_iter_{transfer_flag}-{start_idx}_{max_idx}_{step_idx}.csv"
    get_feature_psm_std(psm_list, train_x, result_save_file)
    my_logger.info("=========================end==========================")


if __name__ == '__main__':

    max_idx = 20
    step_idx = 1
    is_transfer = int(sys.argv[1])
    transfer_flag = "transfer" if is_transfer == 1 else "no_transfer"

    DATA_SOURCE_PATH = f"/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/data/24h/"  # 训练集的X和Y
    PSM_SAVE_PATH = f'/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/result/personal_model_with_xgb/24h_xgb_model/24h_{transfer_flag}_psm/'
    INIT_PSM_PATH = f'/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/result/personal_model_with_xgb/24h_xgb_model/'

    init_psm_file = "0006_xgb_global_feature_weight_boost100.csv"

    my_logger = MyLog().logger

    if is_transfer == 1:
        # 0008_24h_9_feature_weight_gtlboost20_localboost50.csv
        psm_flag = "feature_weight_gtlboost20_localboost50"
    else:
        psm_flag = f"feature_weight_localboost70_{transfer_flag}"

    # print(f"cosine result:")
    get_psm_dist(start_idx=1, max_idx=max_idx, step_idx=step_idx, dist_type="euclidean")
    get_psm_dist(start_idx=1, max_idx=max_idx, step_idx=step_idx, dist_type="cosine")
    get_feature_psm_std_run(start_idx=1, max_idx=max_idx, step_idx=step_idx)


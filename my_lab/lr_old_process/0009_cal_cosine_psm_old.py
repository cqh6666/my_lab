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
    for cur_idx in range(start_idx, max_idx + 1, step_idx):
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

    for cur_idx in range(start_idx, max_idx + 1, step_idx):
        file_flag = f"0008_24h_{cur_idx}_"
        psm_file_name = os.path.join(PSM_SAVE_PATH, f"{file_flag}{psm_flag}.csv")
        cur_weight = pd.read_csv(psm_file_name).squeeze()
        my_logger.warning(f"load {psm_file_name}...")
        all_weight.append(cur_weight)

    if len(all_weight) == 1:
        my_logger.error("no find psm file... stop save result file")
        return

    if dist_type == "cosine":
        save_file_name = os.path.join(TEMP_RESULT_PATH, f"lr_old_psm_{dist_type}_by_iter_{transfer_flag}-{start_idx}_{max_idx}_{step_idx}.csv")
        pd.DataFrame(cosine_similarity(all_weight)).to_csv(save_file_name)
        my_logger.info(f"success save cosine similarity by psm csv! {save_file_name}")
    elif dist_type == "euclidean":
        save_file_name = os.path.join(TEMP_RESULT_PATH, f"lr_old_psm_{dist_type}_by_iter_{transfer_flag}-{start_idx}_{max_idx}_{step_idx}.csv")
        pd.DataFrame(euclidean_distances(all_weight)).to_csv(save_file_name)
        my_logger.info(f"success save euclidean distances by psm csv! {save_file_name}")


if __name__ == '__main__':

    max_idx = 50
    step_idx = 5
    pre_hour = 24
    is_transfer = int(sys.argv[1])
    transfer_flag = "transfer" if is_transfer == 1 else "no_transfer"
    psm_flag = f"psm_{transfer_flag}"

    root_dir = f"{pre_hour}h_old2"
    DATA_SOURCE_PATH = f"/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/data/{root_dir}/"  # 训练集的X和Y
    PSM_SAVE_PATH = f'/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/result/personal_model_with_lr/{root_dir}/psm_{transfer_flag}/'
    INIT_PSM_PATH = f'/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/result/personal_model_with_lr/{root_dir}/global_model/'
    TEMP_RESULT_PATH = f"/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/result/personal_model_with_lr/{root_dir}/temp_result/"

    init_psm_file = f"0006_{pre_hour}h_global_lr_liblinear_400.csv"

    my_logger = MyLog().logger

    get_psm_dist(start_idx=5, max_idx=max_idx, step_idx=step_idx, dist_type="euclidean")
    get_psm_dist(start_idx=5, max_idx=max_idx, step_idx=step_idx, dist_type="cosine")


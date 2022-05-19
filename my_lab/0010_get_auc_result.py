# -*- coding: gbk -*-
"""
-------------------------------------------------
   File Name:     0010_get_auc_result
   Description:   ...
   Author:        cqh
   date:          2022/4/20 20:40
-------------------------------------------------
   Change Activity:
                  2022/4/20:
-------------------------------------------------
"""
__author__ = 'cqh'

import pandas as pd
import os
from sklearn.metrics import roc_auc_score
import numpy as np
import sys
import time
from my_logger import MyLog


def get_concat_result(file_flag):
    """
    :param file_flag: 文件标志，找到同一个迭代次数的所有csv
    :return:
    """
    concat_start = time.time()
    # 遍历文件夹
    all_files = os.listdir(CSV_RESULT_PATH)
    all_result = pd.DataFrame()
    count = 0
    for file in all_files:
        if file_flag in file:
            count += 1
            result = pd.read_csv(os.path.join(CSV_RESULT_PATH, file), index_col=False)
            result['proba'] = result['proba'].str.strip('[]').astype(np.float64)
            all_result = pd.concat([all_result, result], axis=0)
            logger.info(f"load {file} csv and concat success!")
            # 合并后删除
            # os.remove(os.path.join(CSV_RESULT_PATH, file))

    logger.info(f"find {count} csv and all the result shape is: {all_result.shape}")
    logger.info(f"concat time:  {time.time() - concat_start} s")
    if all_result.shape[0] == 0:
        logger.error(f"find no csv result...")
        return None
    try:
        all_result.to_csv(all_result_file)
        logger.warning(f"concat all result to csv {all_result_file} success!")
        return all_result
    except Exception as err:
        logger.error(err)
        raise err


def cal_auc_result():
    """
    读取csv文件，分别存有real和proba列，计算auc结果
    :return:
    """
    result = pd.read_csv(all_result_file)
    y_test = result['real']
    y_pred = result['proba']

    logger.info(f"result shape: {result.shape}")

    score = roc_auc_score(y_test, y_pred)
    result_score_str = f"{learned_metric_iteration},{score}"
    with open(result_file, 'a+') as f:
        f.write(result_score_str + "\n")
        logger.warning(result_score_str)
        logger.warning("cal auc and save success!")


def process_and_plot_result():
    """
    处理auc结果 并绘制成图像
    :return:
    """
    with open(AUC_RESULT_PATH, "r") as f:
        iter_idx = []
        auc = []
        for line in f.readlines():
            line_str = line.split(',')
            iter_idx.append(line_str[0])
            auc.append(line_str[1])

        data = {'iter': iter_idx, 'auc': auc}
        all_auc_result = pd.DataFrame(data)
        all_auc_result['iter'] = all_auc_result['iter'].astype(int)
        all_auc_result['auc'] = all_auc_result['auc'].astype(float)

    all_auc_result.sort_values(by=['iter'], inplace=True)
    ax = all_auc_result.plot(x='iter', y='auc')
    fig = ax.get_figure()
    png_file_name = os.path.join(AUC_RESULT_PATH, f"24h_auc_result_{transfer_flag}.png")
    fig.savefig(png_file_name)
    logger.info("save auc result to png success!")


if __name__ == '__main__':
    learned_metric_iteration = str(sys.argv[1])
    # 1 代表 迁移 ， 0 代表 不迁移
    is_transfer = int(sys.argv[2])

    logger = MyLog().logger

    # 是否迁移，对应不同路径
    transfer_flag = "transfer" if is_transfer == 1 else "no_transfer"
    CSV_RESULT_PATH = f'/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/result/personal_model_with_xgb/24h_xgb_model/24h_test_result_{transfer_flag}/'
    AUC_RESULT_PATH = f'/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/result/personal_model_with_xgb/24h_xgb_model/24h_test_auc_{transfer_flag}/'

    # 根据迭代次数查找到所有的分批量（每1500个）的预测概率csv文件夹
    flag = f"0009_{learned_metric_iteration}_"
    # 多批量整合而成的整体csv文件名
    all_prob_csv_name = f'{flag}all_proba_{transfer_flag}.csv'
    # 将最终的auc结果进行保存
    result_file = os.path.join(AUC_RESULT_PATH, f"24h_auc_result_{transfer_flag}.txt")

    # 先合并再计算auc
    all_result_file = os.path.join(AUC_RESULT_PATH, all_prob_csv_name)
    if os.path.exists(all_result_file):
        logger.warning(f"exist {all_result_file}, will not concat and cal result...")
    else:
        get_concat_result(flag)
        cal_auc_result()

    # ================================ end ===================================
    # 等结果够多才进行绘制图像
    # process_and_plot_result()
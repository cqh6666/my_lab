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


def get_concat_result(file_name, file_flag):
    concat_start = time.time()
    # 遍历文件夹
    all_files = os.listdir(SOURCE_PATH)
    all_result = pd.DataFrame()

    for file in all_files:
        if file_flag in file:
            result = pd.read_csv(os.path.join(SOURCE_PATH, file), index_col=False)
            result['proba'] = result['proba'].str.strip('[]').astype(np.float64)
            all_result = pd.concat([all_result, result], axis=0)
            logger.info(f"the {file_flag} csv saved success!")
            # 合并后删除
            # os.remove(os.path.join(SOURCE_PATH, file))

    logger.info(f"after concat the result shape is: {all_result.shape}")
    logger.info(f"concat time:  {time.time() - concat_start} s")

    try:
        all_result.to_csv(os.path.join(SAVE_PATH, file_name))
        logger.warning(f"concat all result to csv {file_name} success!")
    except Exception as err:
        logger.error(err)
        raise err


def cal_auc_result(file, flag):
    result = pd.read_csv(os.path.join(SAVE_PATH, file))
    y_test = result['real']
    y_pred = result['proba']

    logger.info(f"y_test shape: {y_test.shape}")
    logger.info(f"y_pred shape: {y_pred.shape}")

    logger.warning(f"{flag} auc: {roc_auc_score(y_test, y_pred)}")
    logger.warning("cal success!")


if __name__ == '__main__':
    SOURCE_PATH = '/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/result/personal_model_with_xgb/24h_transfer_xgb_test_result/'
    SAVE_PATH = '/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/result/personal_model_with_xgb/auc/'
    learned_metric_iteration = str(sys.argv[1])

    logger = MyLog().logger

    flag = f"0009_{learned_metric_iteration}"
    # 保存文件名
    file_name = f'{flag}_proba_tran_all.csv'
    # 先合并再计算auc
    get_concat_result(file_name, flag)
    cal_auc_result(file_name, flag)

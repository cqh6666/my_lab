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

SOURCE_PATH = '/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/result/personal_model_with_xgb/24h_transfer_xgb_test_result/'


def get_concat_result(file_name, file_flag):
    concat_start = time.time()
    all_files = os.listdir(SOURCE_PATH)
    all_result = pd.DataFrame()

    for file in all_files:
        if file_flag in file:
            result = pd.read_csv(os.path.join(SOURCE_PATH, file), index_col=False)
            result['proba'] = result['proba'].str.strip('[]').astype(np.float64)
            all_result = pd.concat([all_result, result], axis=0)
            print(f"the {file_flag} csv saved success!")
            # 合并后删除
            os.remove(os.path.join(SOURCE_PATH, file))

    print(all_result.shape)
    all_result.to_csv(os.path.join(SOURCE_PATH, file_name))
    print("concat time: ", time.time() - concat_start)
    print("save all result success!")


def cal_auc_result(file, flag):
    cal_start = time.time()

    result = pd.read_csv(os.path.join(SOURCE_PATH, file))

    y_test = result['real']
    y_pred = result['proba']

    print("y_test shape", y_test.shape)
    print("y_pred shape", y_pred.shape)

    print(f"{flag} auc", roc_auc_score(y_test, y_pred))
    print("cal time: ", time.time() - cal_start)
    print("cal success!")


if __name__ == '__main__':
    # 10,50,115,120
    learned_metric_iteration = str(sys.argv[1])
    flag = f"0009_{learned_metric_iteration}"
    file_name = f'{flag}_all_proba_tran.csv'
    get_concat_result(file_name, flag)
    cal_auc_result(file_name, flag)

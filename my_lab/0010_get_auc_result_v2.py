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

SOURCE_PATH = '/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/result/personal_model_with_xgb/test_result/'
SAVE_PATH = '/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/result/personal_model_with_xgb/'


def get_concat_result(file_name, learned_metric_iteration):
    concat_start = time.time()
    all_files = os.listdir(SOURCE_PATH)
    all_result = pd.DataFrame()

    for file in all_files:
        if learned_metric_iteration in file:
            result = pd.read_csv(os.path.join(SOURCE_PATH, file), index_col=False)
            result['proba'] = result['proba'].str.strip('[]').astype(np.float64)
            all_result = pd.concat([all_result, result], axis=0)

    print(all_result.shape)
    all_result.to_csv(os.path.join(SAVE_PATH, file_name))
    print("concat time: ", time.time() - concat_start)
    print("save success!")


def cal_auc_result(file, learned_metric_iteration):
    cal_start = time.time()

    result = pd.read_csv(os.path.join(SAVE_PATH, file))

    # result['predict'] = np.where(result['proba'] < 0.5, 0, 1)
    # result['proba'] = result['proba'].apply(lambda x: 0 if result['proba'] < 0.5 else 1)

    y_test = result['real']
    y_pred = result['proba']

    print("y_test shape", y_test.shape)
    print("y_pred shape", y_pred.shape)

    print(f"{learned_metric_iteration} auc", roc_auc_score(y_test, y_pred))
    print("cal time: ", time.time() - cal_start)
    print("cal success!")


if __name__ == '__main__':
    # 10,50,115,120
    learned_metric_iteration = int(sys.argv[1])
    file_name = f'0010_auc_result_{learned_metric_iteration}.csv'
    get_concat_result(file_name, learned_metric_iteration)
    cal_auc_result(file_name, learned_metric_iteration)

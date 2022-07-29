# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     model_train
   Description:   ...
   Author:        cqh
   date:          2022/7/21 16:42
-------------------------------------------------
   Change Activity:
                  2022/7/21:
-------------------------------------------------
"""
__author__ = 'cqh'

import os


import pandas as pd

import warnings

from S11_get_output_csv import train_strategy
from utils.data_utils import get_train_test_X_y, get_model_dict, save_to_csv_by_row
from utils.score_utils import get_all_score

warnings.filterwarnings('ignore')


def get_all_model_score(all_model_dict, train_x, train_y, save_path='', index_desc='fit', strategy_select=1):

    for name, clf in all_model_dict.items():
        train_prob, test_prob = train_strategy(clf, strategy_select, train_x, train_y, test_data_x)

        all_score_dict = get_all_score(test_data_y, test_prob, train_prob)
        all_res_df = pd.DataFrame(data=all_score_dict, index=[name + "+" + index_desc])
        all_score_file = os.path.join(save_path, f"all_scores_v{version}.csv")
        save_to_csv_by_row(all_score_file, all_res_df)


if __name__ == '__main__':

    is_engineer = True
    is_norm = False
    print(is_engineer, is_norm)
    train_data_x, test_data_x, train_data_y, test_data_y = get_train_test_X_y(is_engineer, is_norm)
    """
    old: 原始数据                   is_engineer=False, is_norm=False
    old_norm: 原始数据 + 标准化      is_engineer=False, is_norm=True
    new: 新数据                    is_engineer=True, is_norm=False
    new_norm: 新数据 + 标准化       is_engineer=True, is_norm=True
    """
    version = 8
    dir_path = f"new_result_csv/new"

    model_select = ['dt', 'lr', 'rf', 'xgb', 'lgb']
    model_dict = get_model_dict(model_select)

    strategy_selects = [1,  2, 3]
    index_descs = ['fit', 'fit+calibration', 'calibration+fit']
    for select, desc in zip(strategy_selects, index_descs):
        get_all_model_score(all_model_dict=model_dict, save_path=dir_path, strategy_select=select, index_desc=desc)
        print(select, desc, "done!")
    # get_all_model_score(model_dict, dir_path)

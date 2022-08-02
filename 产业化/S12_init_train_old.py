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

from utils.data_utils import get_train_test_X_y, get_model_dict, save_to_csv_by_row
from utils.score_utils import get_all_score
from utils.strategy_utils import train_strategy

warnings.filterwarnings('ignore')


def get_all_model_score(all_model_dict, all_model_desc, save_path='', index_desc='fit', strategy_select=1):
    for name, clf in all_model_dict.items():
        train_prob, test_prob = train_strategy(clf, strategy_select, train_data_x, train_data_y, test_data_x)

        all_score_dict = get_all_score(test_data_y, test_prob, train_prob)
        all_res_df = pd.DataFrame(data=all_score_dict, index=[all_model_desc[name] + index_desc])
        all_score_file = os.path.join(save_path, f"all_scores_v{version}.csv")
        save_to_csv_by_row(all_score_file, all_res_df)


if __name__ == '__main__':

    is_engineer = False
    is_norm = False
    print(is_engineer, is_norm)
    train_data_x, test_data_x, train_data_y, test_data_y = get_train_test_X_y(is_engineer, is_norm)
    """
    old: 原始数据                   is_engineer=False, is_norm=False
    old_norm: 原始数据 + 标准化      is_engineer=False, is_norm=True
    new: 新数据                    is_engineer=True, is_norm=False
    new_norm: 新数据 + 标准化       is_engineer=True, is_norm=True
    """
    version = 15
    dir_path = f"new_result_csv/old/v{version}"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # 初始建模
    init_model_select = ['lr', 'mlp', 'xgb']
    init_model_dict = get_model_dict(init_model_select, engineer=True)
    init_model_desc = {
        'lr': '原始逻辑回归',
        'mlp': '原始MLP',
        'xgb': '原始XGB'
    }
    get_all_model_score(all_model_dict=init_model_dict, all_model_desc=init_model_desc, save_path=dir_path,
                        strategy_select=1, index_desc='')
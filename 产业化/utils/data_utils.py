# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     data_utils
   Description:   ...
   Author:        cqh
   date:          2022/7/27 12:07
-------------------------------------------------
   Change Activity:
                  2022/7/27:
-------------------------------------------------
"""
__author__ = 'cqh'

import json
import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
from sklearn.neural_network import MLPClassifier
import numpy as np

random_state = 2022


class NumpyEncoder(json.JSONEncoder):
    """
    ndarray to list
    """

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def get_train_test_data(engineer=False, norm=False):
    """
    :param norm: 是否进行了标准化
    :param engineer: 是否进行了特征工程
    :return:
    """

    if engineer:
        if norm:
            train_data = pd.read_csv("./data_csv_old/new_data/all_train_data_norm.csv")
            test_data = pd.read_csv("./data_csv_old/new_data/all_test_data_norm.csv")
        else:
            train_data = pd.read_csv("./data_csv_old/new_data/all_train_data.csv")
            test_data = pd.read_csv("./data_csv_old/new_data/all_test_data.csv")
    else:
        if norm:
            train_data = pd.read_csv("./data_csv/raw_data/all_train_data_norm.csv")
            test_data = pd.read_csv("./data_csv/raw_data/all_test_data_norm.csv")
        else:
            train_data = pd.read_csv("./data_csv/raw_data/all_train_data.csv")
            test_data = pd.read_csv("./data_csv/raw_data/all_test_data.csv")

    return train_data, test_data


def get_train_test_X_y(engineer=False, norm=False):
    train_data, test_data = get_train_test_data(engineer, norm)
    train_data_x = train_data.drop(['default payment next month'], axis=1)
    train_data_y = train_data['default payment next month']
    test_data_x = test_data.drop(['default payment next month'], axis=1)
    test_data_y = test_data['default payment next month']
    return train_data_x, test_data_x, train_data_y, test_data_y


def get_model(model_name, engineer=False):
    """
    根据模型名称简写获取模型
    :param engineer: 是否进行了特征工程
    :param model_name: 模型名称
    :return:
    """
    _, xgb_params = get_best_params(engineer)

    return {
        "lr": LogisticRegression(max_iter=500, random_state=random_state),
        "rf": RandomForestClassifier(random_state=random_state),
        "mlp": MLPClassifier(random_state=random_state),
        "xgb": xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", random_state=random_state),
        "lgb": lgb.LGBMClassifier(random_state=random_state),
        "best_xgb": xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", random_state=random_state,
                                      **xgb_params),
    }.get(model_name, None)


def get_best_params(engineer):
    if not engineer:
        xgb_params = {'colsample_bytree': 0.6741473407266286, 'gamma': 4.734790037493003,
                      'learning_rate': 0.07057702530784789, 'max_delta_step': 2.7362679772649434, 'max_depth': 7,
                      'min_child_weight': 8, 'reg_alpha': 1.610251576989095, 'reg_lambda': 0.22578463679880756,
                      'scale_pos_weight': 2.7390526500141794, 'subsample': 0.7496086494361796}
        lgb_params = {'colsample_bytree': 0.8717256051768274, 'learning_rate': 0.08783435745813173, 'max_depth': 5,
                      'min_child_samples': 149, 'min_gain_to_split': 0.42851425223169026, 'n_estimators': 789,
                      'num_leaves': 777, 'reg_alpha': 9.378948217747094, 'reg_lambda': 4.62570403078852,
                      'subsample': 0.7306514568154063}
    else:
        xgb_params = {'colsample_bytree': 0.6300707885237736, 'gamma': 3.853408527747642,
                      'learning_rate': 0.10622104472755567, 'max_delta_step': 3.654179528258079, 'max_depth': 4,
                      'min_child_weight': 2, 'reg_alpha': 0.5486453926100276, 'reg_lambda': 3.6503257743046174,
                      'scale_pos_weight': 4.375463303604714, 'subsample': 0.946007829583101}
        lgb_params = {'colsample_bytree': 0.7563191217183375, 'learning_rate': 0.012317492008644515, 'max_depth': 6,
                      'min_child_samples': 189, 'min_gain_to_split': 0.6284415103373849, 'n_estimators': 742,
                      'num_leaves': 425, 'reg_alpha': 5.999658417103122, 'reg_lambda': 3.8559004715160725,
                      'subsample': 0.6112205042229121}
    return lgb_params, xgb_params


def get_model_dict(model_select=None, engineer=False):
    """
    获取模型字典
    :param engineer:
    :param model_select:
    :return:
    """
    if model_select is None:
        model_select = ['dt', 'lr', 'rf', 'xgb', 'lgb']

    model_dict = {}

    for model_name in model_select:
        model_dict[model_name] = get_model(model_name, engineer)

    return model_dict


def save_to_csv_by_row(csv_file, new_df):
    """
    以行的方式插入csv文件之中，若文件存在则在尾行插入，否则新建一个新的csv；
    :param csv_file: 默认保存的文件
    :param new_df: dataFrame格式 需要包含header
    :return:
    """
    # 保存存入的是dataFrame格式
    assert isinstance(new_df, pd.DataFrame)
    # 不能存在NaN
    if new_df.isna().sum().sum() > 0:
        print("exist NaN...")
        return False

    if os.path.exists(csv_file):
        new_df.to_csv(csv_file, mode='a', index=True, header=False)
    else:
        new_df.to_csv(csv_file, index=True, header=True)

    return True


if __name__ == '__main__':
    m = get_model_dict(['lr', 'xgb'])
    print(m)

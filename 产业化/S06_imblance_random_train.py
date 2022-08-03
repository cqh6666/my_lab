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

import json
import os

import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
import warnings
from sklearn.preprocessing import MinMaxScaler
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
import matplotlib.pyplot as plt

from utils.score_utils import *
from utils.data_utils import MyEncoder
warnings.filterwarnings('ignore')




def get_score(_key, model):
    model.fit(train_x, train_y)
    train_score = model.predict_proba(train_x)[:, 1]
    y_score = model.predict_proba(test_x)[:, 1]
    y_true = test_y.to_numpy()
    auc_info = get_auroc(y_true, y_score)
    prc_info = get_auprc(y_true, y_score)
    gini_info = get_gini(y_true, y_score)
    ks_info = get_ks(y_true, y_score)

    psi_info = get_psi(train_score, y_score)
    result_dict = {
        "auc_info": auc_info,
        "prc_info": prc_info,
        "gini_info": gini_info,
        "ks_info": ks_info,
        "psi_info": psi_info,
    }
    # save
    result_json = json.dumps(result_dict, cls=MyEncoder)

    save_path = f'./all_result_info/{_key}/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_file = os.path.join(save_path, f'model_{_key}_result_info.json')
    with open(save_file, 'w') as f:
        f.write(result_json)


def get_csi_info():
    csi_info = get_csi(train_x, test_x)
    result_dict = {
        "csi_info": csi_info
    }
    result_json = json.dumps(result_dict, cls=MyEncoder)

    save_path = f'./all_result_info/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_file = os.path.join(save_path, f'csi_result_info.json')
    with open(save_file, 'w') as f:
        f.write(result_json)




def get_data(all_data_x, all_data_y, _random_state=2022):

    min_max = MinMaxScaler()
    # 标准化
    norm_array = (all_data_x.abs().max().sort_values(ascending=False) > 100).index
    all_data_x[norm_array] = pd.DataFrame(min_max.fit_transform(all_data_x[norm_array]), columns=norm_array)

    # 采样优化
    pipeline = Pipeline([('over', SMOTETomek(random_state=_random_state)),
                         ('under', RandomUnderSampler(random_state=_random_state))])
    all_data_x, all_data_y = pipeline.fit_resample(all_data_x, all_data_y)

    return train_test_split(all_data_x, all_data_y, test_size=0.3, random_state=_random_state)


if __name__ == '__main__':
    data_source = "./default of credit card clients_new.csv"
    all_data = pd.read_csv(data_source)
    data_x = all_data.drop(['ID', 'default payment next month'], axis=1)
    data_y = all_data['default payment next month']
    random_state = 2022
    train_x, test_x, train_y, test_y = get_data(all_data_x=data_x, all_data_y=data_y, _random_state=random_state)

    model_dict = {
        "lr": LogisticRegression(random_state=random_state),
        "rf": RandomForestClassifier(random_state=random_state),
        "xgb": xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", random_state=random_state),
        "lgb": lgb.LGBMClassifier(random_state=random_state),
    }

    # get all result
    for key, value in model_dict.items():
        get_score(key, value)

    # get csi info
    get_csi_info()

    # plot
    for model_name in model_dict.keys():
        plot_auroc(model_name)
        plot_auprc(model_name)
        plot_ks(model_name)

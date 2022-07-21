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

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

all_data = pd.read_csv("./default of credit card clients.csv")
all_data_x = all_data.drop(['default payment next month'], axis=1)
all_data_y = all_data['default payment next month']

model_list = {
    "dtree": DecisionTreeClassifier(),
    "lr": LogisticRegression(),
    "rf": RandomForestClassifier(),
    "xgb": xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss"),
    "gbm": lgb.LGBMClassifier(),
    "mlp": MLPClassifier(),
    "svc": SVC()
}


def get_auc_score(model_dict, data_x, data_y):
    columns = list(model_dict.keys())
    auc_list = []
    for model in model_dict.values():
        auc = cross_val_score(model, data_x, data_y, scoring="roc_auc", cv=5).mean()
        auc_list.append(auc)

    result_df = pd.DataFrame(data=[auc_list], columns=columns)
    return result_df


result_df = get_auc_score(model_list, all_data_x, all_data_y)
result_df.to_csv("S01_init_auc_result.csv")
print(result_df)

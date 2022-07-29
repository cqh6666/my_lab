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
from sklearn.preprocessing import MinMaxScaler
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
import sys

warnings.filterwarnings('ignore')

all_data = pd.read_csv("data_csv/default of credit card clients_new.csv")
all_data_x = all_data.drop(['default payment next month', 'ID'], axis=1)
all_data_y = all_data['default payment next month']

# min_max = MinMaxScaler()
# numer_list = ['LIMIT_BAL', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
#               'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
# all_data_x[numer_list] = pd.DataFrame(min_max.fit_transform(all_data_x[numer_list]), columns=numer_list)

pipeline = Pipeline([('over', SMOTETomek(random_state=2022)),
                     ('under', RandomUnderSampler(random_state=2022))])
all_data_x, all_data_y = pipeline.fit_resample(all_data_x, all_data_y)

model_list = {
    # "dtree": DecisionTreeClassifier(),
    # "lr": LogisticRegression(),
    "rf": RandomForestClassifier(),
    "xgb": xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss"),
    "gbm": lgb.LGBMClassifier(),
    # "mlp": MLPClassifier(),
    # "svc": SVC()
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
result_df.to_csv(f"S04_SMOTETomek_auc_result_no_norm_v2.csv")
print(result_df)

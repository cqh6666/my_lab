# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     train_best_params_xgb
   Description:   ...
   Author:        cqh
   date:          2022/7/26 14:48
-------------------------------------------------
   Change Activity:
                  2022/7/26:
-------------------------------------------------
"""
__author__ = 'cqh'

from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import xgboost as xgb

all_data = pd.read_csv("data_csv/default of credit card clients_new.csv")
all_data_x = all_data.drop(['default payment next month', 'ID'], axis=1)
all_data_y = all_data['default payment next month']

random_state = 2022
# 标准化
norm_array = (all_data_x.abs().max().sort_values(ascending=False) > 100).index
min_max = MinMaxScaler()
all_data_x[norm_array] = pd.DataFrame(min_max.fit_transform(all_data_x[norm_array]), columns=norm_array)

# SMOTE
pipeline = Pipeline([('over', SMOTETomek(random_state=2022)),
                     ('under', RandomUnderSampler(random_state=2022))])
all_data_x, all_data_y = pipeline.fit_resample(all_data_x, all_data_y)

# 分割数据集
train_x, test_x, train_y, test_y = train_test_split(all_data_x, all_data_y, test_size=0.3, random_state=2022)

best_params = {
    'colsample_bytree': 1.0,
    'gamma': 0.037771688423120904,
    'learning_rate': 0.2,
    'max_delta_step': 4.793079249117746,
    'max_depth': 9,
    'min_child_weight': 1,
    'reg_alpha': 1.2984055962660739,
    'reg_lambda': 1.5666212972904408,
    'scale_pos_weight': 7,
    'subsample': 1.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss"
}

xgb_model = xgb.XGBClassifier(random_state=random_state)
xgb_model.fit(train_x, train_y)
predict_prob = xgb_model.predict_proba(test_x)[:, 1]

auc = roc_auc_score(y_true=test_y, y_score=predict_prob)
print(auc)

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

from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import lightgbm as lgb
import warnings
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')

train_data = pd.read_csv("data_csv/all_train_data.csv")
test_data = pd.read_csv("data_csv/all_test_data.csv")

all_data_x = train_data.drop(['default payment next month'], axis=1)
all_data_y = train_data['default payment next month']

test_data_x = test_data.drop(['default payment next month'], axis=1)
test_data_y = test_data['default payment next month']

# pipeline = Pipeline([('over', SMOTETomek(random_state=2022)),
#                      ('under', RandomUnderSampler(random_state=2022))])
# all_data_x, all_data_y = pipeline.fit_resample(all_data_x, all_data_y)

model_list = {
    "dtree": DecisionTreeClassifier(),
    "lr": LogisticRegression(),
    "rf": RandomForestClassifier(),
    "xgb": xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss"),
    "gbm": lgb.LGBMClassifier(),
    "mlp": MLPClassifier(),
    # "svc": SVC()
}

for name, model in model_list.items():
    model.fit(all_data_x, all_data_y)
    y_predict = model.predict_proba(test_data_x)[:, 1]
    auc = roc_auc_score(test_data_y, y_predict)
    print(name, auc)


def get_auc_score(model_dict, data_x, data_y):
    columns = list(model_dict.keys())
    auc_list = []
    for model in model_dict.values():
        auc = cross_val_score(model, data_x, data_y, scoring="roc_auc", cv=5).mean()
        auc_list.append(auc)

    result_df = pd.DataFrame(data=[auc_list], columns=columns)
    return result_df


# result_df = get_auc_score(model_list, all_data_x, all_data_y)
# result_df.to_csv(f"S04_SMOTETomek_auc_result_v2.csv")
# print(result_df)

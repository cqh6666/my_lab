# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     S06_get_shap_test_data
   Description:   ...
   Author:        cqh
   date:          2022/7/25 19:59
-------------------------------------------------
   Change Activity:
                  2022/7/25:
-------------------------------------------------
"""
__author__ = 'cqh'

import pickle

import pandas as pd
import shap
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings('ignore')


def get_shap_value(_train_x, _test_x):
    model = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss")
    model.fit(_train_x, train_y)
    y_score = model.predict_proba(_test_x)[:, 1]
    auc = roc_auc_score(test_y, y_score)
    explainer = shap.TreeExplainer(model)
    shap_value = explainer.shap_values(_test_x)
    explainer.explain_row()
    res = pd.DataFrame(data=shap_value, columns=_test_x.columns)
    res = res.abs().mean(axis=0)

    # shap_file = f'S06_xgb_shap_weight.csv'
    # pd.DataFrame(res,columns=['shap']).to_csv(shap_file)
    # print("save shap weight success!")
    return auc, res


if __name__ == '__main__':
    all_data_old = pd.read_csv("data_csv/default of credit card clients.csv")
    all_data = pd.read_csv("data_csv/default of credit card clients_new.csv")
    all_data_x = all_data.drop(['default payment next month', 'ID'], axis=1)
    all_data_y = all_data['default payment next month']

    old_columns = all_data_old.columns
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

    i = 0
    remove = None
    columns = ['shape', 'auc', 'remove_columns']
    res_df = pd.DataFrame(columns=columns)
    while i <= 30:
        auc, shap_value_temp = get_shap_value(train_x, test_x)
        res_df = pd.concat([res_df, pd.DataFrame([[train_x.shape[1], auc, remove]], columns=columns)], ignore_index=True, axis=0)
        print("col", train_x.shape[1], "auc", auc, "remove", remove)
        print("========================================")
        shap_value_temp.sort_values(inplace=True)

        index = 0
        while index < train_x.shape[1]:
            remove = shap_value_temp.index[index]
            if remove not in old_columns:
                break
            index += 1

        train_x.drop(remove, inplace=True, axis=1)
        test_x.drop(remove, inplace=True, axis=1)
        i += 1

    res_df.to_csv("S07_根据shap值删除不重要特征(只删除合成特征)-RF.csv")


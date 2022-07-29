# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     S11_get_output_csv
   Description:   ...
   Author:        cqh
   date:          2022/7/27 16:45
-------------------------------------------------
   Change Activity:
                  2022/7/27:
-------------------------------------------------
"""
__author__ = 'cqh'

import os

import shap
from sklearn.calibration import CalibratedClassifierCV
import pandas as pd

from utils.data_utils import get_model_dict, get_train_test_X_y


def get_shap_value(model):
    """
    xgb模型
    """
    model.fit(train_data_x, train_data_y)

    explainer = shap.TreeExplainer(model)
    train_shap = explainer.shap_values(train_data_x)
    train_shap_df = pd.DataFrame(data=train_shap, columns=train_data_x.columns, index=train_data_x.index)
    train_shap_df['shap_sum'] = train_shap_df.sum(axis=1)

    test_shap = explainer.shap_values(test_data_x)
    test_shap_df = pd.DataFrame(data=test_shap, columns=test_data_x.columns, index=test_data_x.index)
    test_shap_df['shap_sum'] = test_shap_df.sum(axis=1)

    return train_shap_df, test_shap_df


def train_strategy(clf, strategy=3):
    """
    不同策略进行训练
    策略0： 平衡，不校准
    策略1： 平衡，fit后校准
    策略2:  平衡，校准后fit
    :param clf:
    :param strategy: 1, 2, 3
    :return:
    """
    if strategy == 1:
        pass
    elif strategy == 2:
        clf.fit(train_norm_x, train_norm_y)
        clf = CalibratedClassifierCV(clf, cv="prefit")
    elif strategy == 3:
        clf = CalibratedClassifierCV(clf, cv=5)
    else:
        raise ValueError("Not found strategy!")
    clf.fit(train_norm_x, train_norm_y)
    train_prob = clf.predict_proba(train_norm_x)[:, 1]
    test_prob = clf.predict_proba(test_norm_x)[:, 1]
    return train_prob, test_prob


def get_output_file(model_name, model):
    """
    输出4个文件
    :param model_name: 模型名称
    :param model: 模型
    :return:
    """
    save_path = f"./S11_result/{model_name}/v{version}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print("create new path", save_path)

    # 获得训练集测试集预测概率
    train_pred, test_pred = train_strategy(model)

    train_pred_df = pd.DataFrame(train_pred, index=train_data_x.index, columns=['predict_prob'])
    test_pred_df = pd.DataFrame(test_pred, index=test_data_x.index, columns=['predict_prob'])
    # 1. 训练集：X,Y,预测概率（calibration+fit）
    train_output_df = pd.concat([train_data_x, train_data_y, train_pred_df], axis=1)
    train_output_df.to_csv(os.path.join(save_path, f"train_data_output.csv"), index=False)

    # 2. 测试集：X,Y,预测概率（calibration+fit）
    test_output_df = pd.concat([test_data_x, test_data_y, test_pred_df], axis=1)
    test_output_df.to_csv(os.path.join(save_path, f"test_data_output.csv"), index=False)

    # 3. 训练集：X_SHAP（fit）,Y,预测概率（calibration+fit）,预测总分（fit）
    train_shap_df, test_shap_df = get_shap_value(model)
    train_shap_df['predict_prob'] = train_pred
    train_shap_df['y_true'] = train_data_y
    train_shap_df.to_csv(os.path.join(save_path, f"train_shap_output.csv"), index=False)

    # 4. 测试集：X_SHAP（fit）,Y, 预测概率,（calibration+fit） 预测总分（fit）
    test_shap_df['predict_prob'] = test_pred
    test_shap_df['y_true'] = test_data_y
    test_shap_df.to_csv(os.path.join(save_path, f"test_shap_output.csv"), index=False)


if __name__ == '__main__':
    is_engineer = True
    is_norm = False
    print(is_engineer, is_norm)
    # 原始数据(作为展示)
    train_data_x, test_data_x, train_data_y, test_data_y = get_train_test_X_y(is_engineer, is_norm)

    is_norm = True
    print("is_norm", is_norm)
    # 处理过后的数据(用来作训练)
    train_norm_x, test_norm_x, train_norm_y, test_norm_y = get_train_test_X_y(is_engineer, is_norm)

    random_state = 2022
    version = 2

    # xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", random_state=random_state, **xgb_params),
    models = ['xgb']
    model_list = get_model_dict(model_select=models, engineer=True)
    best_model = model_list.get('xgb')

    # 传入名称和最好的模型
    get_output_file("xgb", best_model)

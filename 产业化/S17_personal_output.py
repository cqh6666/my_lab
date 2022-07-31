#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 17:15:13 2022

@author: liukang
"""
import json
import os

import numpy as np
import pandas as pd

test_data = pd.read_csv("output_json/input_csv/pred_raw_data/test_data_output.csv")
test_shap = pd.read_csv("output_json/input_csv/pred_raw_data/test_shap_output.csv")

first_X_feature_name = 'LIMIT_BAL'
last_X_feature_name = 'PAY_AMT6'
Label_name = 'default payment next month'
shap_sum_name = 'shap_sum'
prediction_name = 'predict_prob'
personal_value_feature_name = 'BILL_AMT1'
num_of_show_feature = 6

all_test_sample_X_shap = test_shap.loc[:, first_X_feature_name:last_X_feature_name]
all_test_sample_mean_shap = all_test_sample_X_shap.mean(axis=0)

# 测试样例前5个
test_example_x = test_data.iloc[:5].drop([Label_name, prediction_name], axis=1)

# 获取中文属性名
all_data = pd.read_csv("data_csv/default of credit card clients_new(Chinese).csv", encoding='gbk')
all_data = all_data.drop(['ID', Label_name], axis=1)
new_columns = all_data.columns.tolist()

old_columns = test_data.columns.tolist()
old_columns.remove(Label_name)
old_columns.remove(prediction_name)
columns_dict = dict(zip(old_columns, new_columns))

# 对应的ids
test_example_x_ids = test_example_x.index.tolist()

version = 12

def getDataExampleInfo(data_samples):
    """
    选取5行样例
    :param data_samples: dataFrame格式
    :return: list格式
    """

    assert isinstance(data_samples, pd.DataFrame)
    # data_samples.columns = new_columns
    data_samples['ID'] = data_samples.index
    return data_samples.to_dict(orient='records')


def personal_analysis_info(target_id):
    target_sample_X_shap = test_shap.loc[target_id, first_X_feature_name:last_X_feature_name]
    shap_diff_between_target_and_general = target_sample_X_shap - all_test_sample_mean_shap
    abs_shap_diff = shap_diff_between_target_and_general.abs()
    abs_shap_diff.sort_values(ascending=False, inplace=True)
    person_important_feature = abs_shap_diff.index.tolist()[:num_of_show_feature]

    # output_1: 预测概率
    person_ori_proba = test_shap.loc[target_id, prediction_name]

    other_sample_proba_lower_than_target_true = test_shap.loc[:, prediction_name] < person_ori_proba
    other_sample_proba_lower_than_target_select = test_shap.loc[other_sample_proba_lower_than_target_true]

    # outper_2: 用户排名
    person_rank = '{}%'.format(
        round((other_sample_proba_lower_than_target_select.shape[0] / test_shap.shape[0] * 100), 1))

    # output_3: 个人重要性排名
    show_target_feature_data = pd.DataFrame()

    person_ori_sum_shap = np.log(person_ori_proba / (1 - person_ori_proba))

    for i in range(num_of_show_feature):
        show_target_feature_data.loc[person_important_feature[i], 'value'] = test_data.loc[
            target_id, person_important_feature[i]]
        show_target_feature_data.loc[person_important_feature[i], 'shap'] = test_shap.loc[
            target_id, person_important_feature[i]]

        other_shap_lower_than_target_true = test_shap.loc[:, person_important_feature[i]] < test_shap.loc[
            target_id, person_important_feature[i]]
        other_shap_lower_than_target_select = test_shap.loc[other_shap_lower_than_target_true]
        show_target_feature_data.loc[person_important_feature[i], 'shap_rank'] = '{}%'.format(
            round((other_shap_lower_than_target_select.shape[0] / test_shap.shape[0] * 100), 1))

        select_shap_diff = shap_diff_between_target_and_general[person_important_feature[i]]
        person_shap_after_del_diff = person_ori_sum_shap - select_shap_diff
        new_person_proba = 1 / (1 + np.exp(-person_shap_after_del_diff))
        show_target_feature_data.loc[person_important_feature[i], 'proba_change'] = '{}%'.format(
            round((person_ori_proba - new_person_proba) * 100, 1))

    show_target_feature_data.to_csv(f"./output_json/input_csv/analysis_data/personal_{id_}_output.csv")

    important_feature_list = []
    rank = 1
    for feature_name in person_important_feature:
        feature_important_info = {
            "rank": rank,
            "name": feature_name,
            "chinaName": columns_dict[feature_name],
            "value": show_target_feature_data.loc[feature_name, 'value'],
            "shap": round(show_target_feature_data.loc[feature_name, 'shap'], 4),
            "shapRank": show_target_feature_data.loc[feature_name, 'shap_rank'],
            "influence": show_target_feature_data.loc[feature_name, 'proba_change']
        }
        rank += 1
        important_feature_list.append(feature_important_info)

    return {
        "预测概率": round(person_ori_proba, 4),
        "用户排名": person_rank,
        "特征分析": important_feature_list
    }


personal_analysis_info_list = []
for id_ in test_example_x_ids:
    personal_analysis_info_list.append(personal_analysis_info(id_))

person_output = {
    "dataExample": getDataExampleInfo(test_example_x),
    "analysisResult": personal_analysis_info_list
}

# save
result_json = json.dumps(person_output, ensure_ascii=False)
mpf_save_file = os.path.join(f'./output_json/v{version}/personalDataInfo.json')
with open(mpf_save_file, 'w', encoding="utf8") as f:
    f.write(result_json)

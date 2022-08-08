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

from utils.data_utils import pretty_floats


def getDataExampleInfo(data_samples):
    """
    选取5行样例 增加ID列
    :param data_samples: dataFrame格式
    :return: list格式
    """

    assert isinstance(data_samples, pd.DataFrame)
    # data_samples.columns = new_columns
    data_samples['ID'] = data_samples.index
    return data_samples.to_dict(orient='records'), data_samples.columns.tolist()


def getDataColsInfo(data_columns):
    """
    传入特征列表
    :param data_columns:
    :return: list格式
    """
    cur_list = []
    for column in data_columns:
        cur_list.append({
            'label': column,
            'prop': column
        })

    return cur_list


def personal_analysis_info(target_id):
    target_sample_X_shap = test_shap.loc[target_id, first_X_feature_name:last_X_feature_name]
    shap_diff_between_target_and_general = target_sample_X_shap - all_test_sample_mean_shap
    abs_shap_diff = shap_diff_between_target_and_general.abs()
    abs_shap_diff.sort_values(ascending=False, inplace=True)
    person_important_feature = abs_shap_diff.index.tolist()[:num_of_show_feature]
    person_important_feature_value = abs_shap_diff[person_important_feature].tolist()

    # output_1: 预测概率
    person_ori_proba = test_shap.loc[target_id, prediction_name]

    other_sample_proba_lower_than_target_true = test_shap.loc[:, prediction_name] < person_ori_proba
    other_sample_proba_lower_than_target_select = test_shap.loc[other_sample_proba_lower_than_target_true]

    # outper_2: 用户排名
    person_rank = '{}%'.format(
        round((other_sample_proba_lower_than_target_select.shape[0] / test_shap.shape[0] * 100), 2))

    # output_3: 个人重要性排名
    show_target_feature_data = pd.DataFrame()

    for i in range(num_of_show_feature):
        show_target_feature_data.loc[person_important_feature[i], 'value'] = test_data.loc[
            target_id, person_important_feature[i]]
        show_target_feature_data.loc[person_important_feature[i], 'shap'] = test_shap.loc[
            target_id, person_important_feature[i]]

        other_shap_lower_than_target_true = test_shap.loc[:, person_important_feature[i]] < test_shap.loc[
            target_id, person_important_feature[i]]
        other_shap_lower_than_target_select = test_shap.loc[other_shap_lower_than_target_true]
        show_target_feature_data.loc[person_important_feature[i], 'shap_rank'] = '{}%'.format(
            round((other_shap_lower_than_target_select.shape[0] / test_shap.shape[0] * 100), 2))

        show_target_feature_data.loc[person_important_feature[i], 'influence'] = person_important_feature_value[i]

    show_target_feature_data.to_csv(f"./output_json/input_csv/analysis_data/personal_{target_id}_output.csv")

    important_feature_list = []
    rank = 1
    for feature_name in person_important_feature:
        feature_important_info = {
            "rank": rank,
            "name": feature_name,
            "chinaName": columns_dict[feature_name],
            "value": show_target_feature_data.loc[feature_name, 'value'],
            "shap": show_target_feature_data.loc[feature_name, 'shap'],
            "shapRank": show_target_feature_data.loc[feature_name, 'shap_rank'],
            "influence": show_target_feature_data.loc[feature_name, 'influence']
        }
        rank += 1
        important_feature_list.append(feature_important_info)

    return {
        "预测概率": person_ori_proba,
        "违约概率排名": person_rank,
        "特征分析": important_feature_list,
    }


def belongToGroup(prob):
    # 不同客群分组概率和对应的名称
    group_range_name = ['优质客户', '良好客户', '普通客户', '较高风险客户', '高风险客户']
    group_range_list = [
        [0, 0.05],
        [0.05, 0.1],
        [0.1, 0.25],
        [0.25, 0.5],
        [0.5, 1]
    ]
    belongIndex = 0
    for group_idx in range(len(group_range_list)):
        cur_group = group_range_list[group_idx]
        if cur_group[0] <= prob <= cur_group[1]:
            belongIndex = group_idx
            break

    return group_range_name[belongIndex]


def get_all_ids_info():
    test_example_x_ids = test_example_x.index.tolist()

    stat_analysis = []
    stat_feature = []
    for test_id in test_example_x_ids:
        cur_info = personal_analysis_info(test_id)

        # 统计分析
        num_list = [cur_info['预测概率'], cur_info['违约概率排名'], belongToGroup(cur_info['预测概率'])]
        num_list_des = ['预测概率', '违约概率排名', '分组']
        stat_analysis.append(pd.DataFrame({"num":num_list,"des":num_list_des}).to_dict(orient='records'))

        # 特征分析
        stat_feature.append(cur_info['特征分析'])

    dataExample, columns = getDataExampleInfo(test_example_x)
    person_output = {
        "dataExample": dataExample,
        "dataCols": getDataColsInfo(data_columns=columns)
    }

    return {
        "统计分析": stat_analysis,
        "特征分析": stat_feature,
        "客户样例": person_output
    }


if __name__ == '__main__':
    test_data = pd.read_csv("output_json/input_csv/best_model/test_data_output.csv")
    test_shap = pd.read_csv("output_json/input_csv/best_model/test_shap_output.csv")

    first_X_feature_name = 'LIMIT_BAL'
    last_X_feature_name = 'PAY_AMT_CHANGE_3/BILL_AMT_CHANGE_3'
    Label_name = 'default payment next month'
    shap_sum_name = 'shap_sum'
    prediction_name = 'predict_prob'
    personal_value_feature_name = 'BILL_AMT1'
    num_of_show_feature = 6

    all_test_sample_X_shap = test_shap.loc[:, first_X_feature_name:last_X_feature_name]
    all_test_sample_mean_shap = all_test_sample_X_shap.mean(axis=0)

    # 测试样例前5个
    test_example_x = test_data.iloc[:5].drop([Label_name, prediction_name], axis=1)

    # 属性名映射
    column_csv = pd.read_csv("data_csv/feature_name.csv", encoding='utf-8')
    columns_dict = dict(zip(column_csv['特征名称'], column_csv['特征解释']))



    version = 18
    # save
    result_json = json.dumps(pretty_floats(get_all_ids_info()), ensure_ascii=False)
    mpf_save_file = os.path.join(f'./output_json/v{version}/CustomerRiskAssessment.json')
    with open(mpf_save_file, 'w', encoding="utf8") as f:
        f.write(result_json)

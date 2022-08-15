#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 09:18:38 2022

@author: liukang
"""
import json
import os
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from utils.data_utils import MyEncoder
from utils.data_utils import pretty_floats


def split_data(data_samples, select_num, rand_state):
    """
    随机抽样
    :param data_samples:
    :param select_num:
    :return:
    """
    random.seed(rand_state)
    random_idx = random.sample(list(range(data_samples.shape[0])), select_num)
    return random_idx


def get_stat_analysis(group_data):
    # 总人数
    person_num = group_data.shape[0]
    # 总体违约率
    avg_prediction = np.mean(group_data[prediction_name])
    # 总体违约人数
    expected_positive = int(group_data.shape[0] * avg_prediction)
    # 总体违约金额
    person_expected_value = group_data[personal_value_feature_name] * group_data[prediction_name]
    total_expect_value = np.sum(person_expected_value.values)

    num_data = [person_num, expected_positive, total_expect_value, '{}%'.format(round((avg_prediction * 100), 2))]
    des_data = ["总人数", "违约人数", "违约金额", "违约率"]
    res_df = pd.DataFrame({"num": num_data, "des": des_data})
    return res_df.to_dict(orient='records')


def get_group_stat_card(group_data_record):
    target_columns = ['type', 'num', 'radio', 'score']
    res_df = group_data_record[target_columns]
    return res_df.to_dict(orient='records')


def get_group_stat_feature(group_data_record, group_important_feature_list):
    # 多个客群
    res_df_list = []
    for group_idx, group_important_feature in zip(range(group_data_record.shape[0]), group_important_feature_list):
        # 单个客群
        res_df = pd.DataFrame()
        for name in group_important_feature:
            res_df.loc[name, "name"] = name
            res_df.loc[name, "chinaName"] = columns_dict[name]
            res_df.loc[name, "val"] = group_data_record.loc[group_idx, f"{name}_val"]
            res_df.loc[name, "influence"] = group_data_record.loc[group_idx, f"{name}_influence"]
        res_df_list.append(res_df.to_dict(orient='records'))
    return res_df_list


def get_stat_pie(group_data_record):
    nums_count = group_data_record['num'].sum()
    cur_list = []
    for group_idx in range(group_data_record.shape[0]):
        group_name = group_data_record.loc[group_idx, 'type']
        group_num = group_data_record.loc[group_idx, 'num']
        cur_dict = {
            "columns": [group_name, '总数'],
            "rows": [
                {group_name: group_name, '总数': group_num},
                {group_name: '其他', '总数': nums_count - group_num}
            ]
        }
        cur_list.append(cur_dict)
    return cur_list


def get_group_stat_radar(group_radar_record):
    return {
        "columns": group_radar_record.columns.tolist(),
        "rows": group_radar_record.to_dict(orient='records')
    }


def get_group_all_info(group_name, group_data, group_shap_data):
    """
    客户群体画像
    :param group_name:
    :param group_data:
    :param group_shap_data:
    :return:
    """
    # calculate data in total test set

    # 最重要的k个特征
    mean_train_shap = train_shap.loc[:, first_X_feature_name:last_X_feature_name].abs().mean(axis=0)
    mean_train_shap.sort_values(ascending=False, inplace=True)
    important_feature = mean_train_shap.index.tolist()[:num_of_radar_feature]
    # important_feature_value = mean_train_shap[important_feature].tolist()

    """
    num 客户人数
    share 客户占比
    特征均值，特征shap均值，特征
    """
    group_idx = 0
    # output_4: 客户群体画像
    group_data_record = pd.DataFrame()
    group_radar_record = pd.DataFrame()

    # 客群重要特征列表
    group_important_feature_list = []
    # 遍历群组字典,key代表名字,value代表概率范围
    for prob_name, prob_range in group_range_dict.items():

        low_threshold = prob_range[0]
        high_threshold = prob_range[1]
        # 每个群组对应名称
        group_data_record.loc[group_idx, 'type'] = prob_name
        # 每个群组概率范围
        group_data_record.loc[group_idx, 'score'] = '{}%-{}%'.format(
            int(low_threshold * 100), int(high_threshold * 100)
        )

        # 每个群组所有客户的原始数据
        sample_in_threshold_true = (group_data.loc[:, prediction_name] > low_threshold) \
                                 & (group_data.loc[:, prediction_name] <= high_threshold)
        sample_in_threshold_select = group_data.loc[sample_in_threshold_true]

        # 每个群组用户人数
        group_data_record.loc[group_idx, 'num'] = sample_in_threshold_select.shape[0]
        # 每个群体占总人数的占比
        group_data_record.loc[group_idx, 'radio'] = \
            '{}%'.format(round((sample_in_threshold_select.shape[0] / group_data.shape[0] * 100), 2))

        # 每个客群所有客户的shap值
        shap_in_threshold_select = group_shap_data.loc[sample_in_threshold_true]
        shap_in_threshold_select.reset_index(drop=True, inplace=True)

        # 计算出每个群组top 6个重要特征
        mean_group_shap = shap_in_threshold_select.loc[:, first_X_feature_name:last_X_feature_name].abs().mean(axis=0)
        mean_group_shap.sort_values(ascending=False, inplace=True)
        group_important_feature = mean_group_shap.index.tolist()[:num_of_show_feature]
        group_important_feature_value = mean_group_shap[group_important_feature].tolist()

        sample_in_threshold_select_important_feature = sample_in_threshold_select.loc[:, group_important_feature]
        important_feature_mean = sample_in_threshold_select_important_feature.mean(axis=0)

        # 存入各群组重要特征列表
        group_important_feature_list.append(group_important_feature)

        # 对于每个群组的重要特征处理
        for j in range(num_of_show_feature):
            group_data_record.loc[group_idx, '{}_val'.format(group_important_feature[j])] = important_feature_mean[j]
            group_data_record.loc[group_idx, '{}_influence'.format(group_important_feature[j])] = group_important_feature_value[j]

        # 雷达图数据分析
        # 根据训练集分析出最重要的k个特征 如上 important_feature
        # 分析每个客群这个特征的shap排名
        group_radar_record.loc[group_idx, 'name'] = prob_name[:-2]
        for feature in important_feature:
            # shap特征排名
            shap_feature_lower = mean_group_shap[mean_group_shap < mean_group_shap[feature]].size
            shap_feature_lower_rank = round(shap_feature_lower / mean_group_shap.size * 100, 2)
            # 影响度排名
            # shap_feature_lower = mean_group_shap[feature] / mean_group_shap.sum() * 100
            group_radar_record.loc[group_idx, columns_dict[feature]] = shap_feature_lower_rank

        group_idx += 1

    # 保存为csv
    group_data_record.to_csv(os.path.join(save_path, f"{group_name}_statistics_output.csv"))

    # 测试集统计分析
    stat_analysis = get_stat_analysis(group_data)

    # 客户分组 饼状图
    stat_pie = get_stat_pie(group_data_record)

    # 客户分组 统计卡片
    stat_card = get_group_stat_card(group_data_record)

    # 客户分组 特征分析
    stat_feature = get_group_stat_feature(group_data_record, group_important_feature_list)

    # 客群雷达图
    stat_radar = get_group_stat_radar(group_radar_record)

    return stat_analysis, stat_pie, stat_card, stat_feature, stat_radar


if __name__ == '__main__':
    test_data = pd.read_csv("output_json/input_csv/best_model/test_data_output.csv")
    test_shap = pd.read_csv("output_json/input_csv/best_model/test_shap_output.csv")
    train_data = pd.read_csv("output_json/input_csv/best_model/train_data_output.csv")
    train_shap = pd.read_csv("output_json/input_csv/best_model/train_shap_output.csv")

    first_X_feature_name = 'LIMIT_BAL'
    last_X_feature_name = 'PAY_AMT_CHANGE_3/BILL_AMT_CHANGE_3'
    Label_name = 'default payment next month'
    # shap_sum_name = 'shap_sum'
    prediction_name = 'predict_prob'
    personal_value_feature_name = 'BILL_AMT1'

    # top k个重要特征
    num_of_show_feature = 6
    num_of_radar_feature = 5
    version = 19

    save_path = f"./output_json/v{version}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 训练集分割部分
    num_of_test_data = 3
    select_num = test_data.shape[0] // num_of_test_data

    # 属性名映射
    column_csv = pd.read_csv("data_csv/feature_name.csv", encoding='utf-8')
    columns_dict = dict(zip(column_csv['特征名称'], column_csv['特征解释']))

    # 不同客群分组概率和对应的名称
    group_range_name = ['优质客户', '良好客户', '普通客户', '较高风险客户', '高风险客户']
    group_range_list = [
        [0, 0.05],
        [0.05, 0.1],
        [0.1, 0.25],
        [0.25, 0.5],
        [0.5, 1]
    ]
    group_range_dict = dict(zip(group_range_name, group_range_list))

    stat_analysis_list = []
    stat_pie_list = []
    stat_card_list = []
    stat_feature_list = []
    stat_radar_list = []

    cur_idx = 1
    while cur_idx <= num_of_test_data:
        random_idx = split_data(test_data, select_num, cur_idx)
        cur_data = test_data.loc[random_idx]
        cur_shap_data = test_shap.loc[random_idx]
        cur_data.reset_index(inplace=True, drop=True)
        cur_shap_data.reset_index(inplace=True, drop=True)
        statistic_analysis, statistic_pie, statistic_card, statistic_feature, statistic_radar = get_group_all_info(
            f"group{cur_idx}", cur_data, cur_shap_data)

        stat_analysis_list.append(statistic_analysis)
        stat_pie_list.append(statistic_pie)
        stat_card_list.append(statistic_card)
        stat_feature_list.append(statistic_feature)
        stat_radar_list.append(statistic_radar)
        cur_idx += 1

    all_group_info = {
        "测试集统计分析": stat_analysis_list,
        "客户分组饼状图": stat_pie_list,
        "客户分组统计卡片": stat_card_list,
        "客户分组特征分析": stat_feature_list,
        "客户分组雷达图": stat_radar_list
    }
    result_json = json.dumps(pretty_floats(all_group_info), cls=MyEncoder, ensure_ascii=False)
    mpf_save_file = os.path.join(save_path, f'CustomerGroupPortrait.json')
    with open(mpf_save_file, 'w', encoding="utf8") as f:
        f.write(result_json)

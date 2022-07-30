# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     get_all_info
   Description:   ...
   Author:        cqh
   date:          2022/7/28 17:08
-------------------------------------------------
   Change Activity:
                  2022/7/28:
-------------------------------------------------
"""
__author__ = 'cqh'

import json
import os

import pandas as pd
import numpy as np

def getTaskIntroductionInfo():
    """
    :return: 字符串格式
    """
    introduction = "This dataset contains information on default payments, demographic factors, credit data, history of payment, and bill statements of credit card clients in Taiwan from April 2005 to September 2005."
    return introduction


def getDataIntroductionInfo(data_columns):
    """
    传入特征列表
    :param data_columns:
    :return: dict格式
    """
    cur_dict = {
        '数据名称': 'Default of Credit Card Clients Dataset',
        '数据来源': 'UCI Machine Learning Repository',
        '特征数': len(data_columns),
        '备注': 'Default Payments of Credit Card Clients in Taiwan from 2005',
        '数据地址': 'https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients'
    }

    return {
        'dataIntroductionCols': list(cur_dict.keys()),
        'dataIntroduction': list(cur_dict.values())
    }


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


def getDataExampleInfo(data_samples):
    """
    选取5行样例
    :param data_samples: dataFrame格式
    :return: list格式
    """
    assert isinstance(data_samples, pd.DataFrame)
    return data_samples.to_dict(orient='records')


def getDataOverviewInfo(data_samples):
    """
    数据集统计信息
    :return: list格式
    """
    assert isinstance(data_samples, pd.DataFrame)

    all_count = data_samples.shape[0]

    #oupput_1: 总体违约率
    avg_prediction = np.mean(data_samples[prediction_name])

    #output_2: 总体违约人数
    expected_positive = int(data_samples.shape[0] * avg_prediction)

    person_expected_value = data_samples[personal_value_feature_name] * data_samples[prediction_name]

    #output_3: 总体违约金额
    total_expect_value = np.sum(person_expected_value.values)

    return [
        {"num": all_count, "des": '总人数'},
        {"num": expected_positive, "des": '违约人数'},
        {"num": round(total_expect_value, 4), "des": "违约金额"},
        {"num": round(avg_prediction, 4), "des": "违约率"},
        {"num": expected_positive, "des": "风险人数"},
        {"num": round(avg_prediction, 4), "des": "风险人数比例"},
        {"num": all_count - expected_positive, "des": "优质人数"},
        {"num": 1 - round(avg_prediction, 4), "des": "优质人数比例"},
    ]


def getModelPerformance(model_score_df):
    """
    模型性能指标
    :param
    ------------------
    model_score_df:
        dataFrame格式
        ['model', 'AUC', ...]

    :return: {columns: .., "rows":...}
    """
    assert isinstance(model_score_df, pd.DataFrame)

    return {
        "columns": model_score_df.columns.to_list(),
        "rows": model_score_df.to_dict(orient='records')
    }


def saveAllStatistics(train_d, test_d, origin_data_x):

    cols = origin_data_x.columns.to_list()

    dataOverview = {
        "trainDataOverview": getDataOverviewInfo(train_d),
        "testDataOverview": getDataOverviewInfo(test_d)
    }

    # 测试样例前5个
    test_example_x = test_d.iloc[:5].drop([Label_name, prediction_name], axis=1)

    all_res_json = {
        "taskIntroductionInfo": getTaskIntroductionInfo(),
        "dataIntroductionInfo": getDataIntroductionInfo(cols),
        "dataColsInfo": getDataColsInfo(cols),
        "dataExample": getDataExampleInfo(test_example_x),
        "dataOverview": dataOverview,
    }
    # save
    result_json = json.dumps(all_res_json, ensure_ascii=False)
    mpf_save_file = os.path.join(save_path, f'allStatisticsInfo.json')
    with open(mpf_save_file, 'w', encoding="utf8") as f:
        f.write(result_json)

    return all_res_json


def saveModelPerformance(score_df):

    mpf = getModelPerformance(score_df)
    # save
    result_json = json.dumps(mpf, ensure_ascii=False)
    mpf_save_file = os.path.join(save_path, f'modelPerformanceInfo.json')
    with open(mpf_save_file, 'w', encoding="utf8") as f:
        f.write(result_json)


if __name__ == '__main__':

    first_X_feature_name = 'LIMIT_BAL'
    last_X_feature_name = 'PAY_AMT6'
    Label_name = 'default payment next month'
    # shap_sum_name = 'shap_sum'
    prediction_name = 'predict_prob'
    personal_value_feature_name = 'BILL_AMT1'
    group_num = 4
    num_of_show_feature = 6

    version = 1
    save_path = f'./v{version}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    all_score = pd.read_csv("output_json/input_csv/all_scores_v7.csv", index_col=0)

    test_data = pd.read_csv("output_json/input_csv/pred_raw_data/test_data_output.csv")
    test_shap = pd.read_csv("output_json/input_csv/pred_raw_data/test_shap_output.csv")
    train_data = pd.read_csv("output_json/input_csv/pred_raw_data/train_data_output.csv")
    train_shap = pd.read_csv("output_json/input_csv/pred_raw_data/train_shap_output.csv")

    origin_test_data_x = test_data.drop([prediction_name, Label_name], axis=1)

    # 保存模型性能信息
    all_score.reset_index(inplace=True)
    saveModelPerformance(all_score)
    # 保存相关统计信息
    saveAllStatistics(train_data, test_data, origin_test_data_x)

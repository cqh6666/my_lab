# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     get_result_info
   Description:   ...
   Author:        cqh
   date:          2022/7/22 18:04
-------------------------------------------------
   Change Activity:
                  2022/7/22:
-------------------------------------------------
"""
__author__ = 'cqh'

import json
import os

from utils.data_utils import get_model_dict, get_train_test_X_y, MyEncoder
from utils.plot_utils import *
from utils.score_utils import get_all_info
import pandas as pd
import numpy as np
from utils.strategy_utils import train_strategy


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

    # 获取所有属性名
    columns = model_score_df.columns.to_list()

    # 用来保存所有指标
    scores_list = []
    for column in columns:
        cur_model_score_df = model_score_df[column].dropna()
        temp_df = pd.DataFrame(data={
            "model": cur_model_score_df.index,
            column: cur_model_score_df
        })

        temp_dict = {
            "columns": temp_df.columns.to_list(),
            "rows": temp_df.to_dict(orient='records')
        }
        scores_list.append(temp_dict)

    return scores_list


# 测试集总览
def getDataOverviewInfo(data_samples):
    """
    数据集统计信息
    :return: list格式
    """
    assert isinstance(data_samples, pd.DataFrame)

    all_count = data_samples.shape[0]

    # output_2: 总体违约人数
    expected_positive = (data_samples[Label_name] == 1).sum()

    # oupput_1: 总体违约率
    avg_prediction = round(expected_positive / all_count, 4)

    # output_3: 总体违约金额
    person_expected_value = data_samples[personal_value_feature_name] * data_samples[Label_name]
    total_expect_value = np.sum(person_expected_value.values)

    return [
        {"num": all_count, "des": '总人数'},
        {"num": expected_positive, "des": '违约人数'},
        {"num": total_expect_value, "des": "违约金额"},
        {"num": avg_prediction, "des": "违约率"},
    ]


"""
获取所有结果信息
"""
test_data = pd.read_csv("output_json/input_csv/best_model/test_data_output.csv")
test_shap = pd.read_csv("output_json/input_csv/best_model/test_shap_output.csv")
train_data = pd.read_csv("output_json/input_csv/best_model/train_data_output.csv")
train_shap = pd.read_csv("output_json/input_csv/best_model/train_shap_output.csv")

first_X_feature_name = 'LIMIT_BAL'
last_X_feature_name = 'PAY_AMT6'
Label_name = 'default payment next month'
# shap_sum_name = 'shap_sum'
prediction_name = 'predict_prob'
personal_value_feature_name = 'BILL_AMT1'
group_num = 4
num_of_show_feature = 6

version = 18

# 初始建模
my_best_model_select = ['best_xgb']
best_model_dict = get_model_dict(my_best_model_select, engineer=True)
clf = best_model_dict.get('best_xgb')

is_engineer = True
is_norm = True
train_data_x, test_data_x, train_data_y, test_data_y = get_train_test_X_y(is_engineer, is_norm)

train_prob, test_prob = train_strategy(clf, 3, train_data_x, train_data_y, test_data_x)

# 绘制曲线坐标
all_res = get_all_info(test_data_y, test_prob, train_prob)
all_score = pd.read_csv(f"output_json/v{version}/all_scores_v{version}.csv", index_col=0)

res_json = {
    'dataOverview': getDataOverviewInfo(test_data),
    'performaceCurve': all_res,
    'performance': getModelPerformance(all_score)
}

# save
result_json = json.dumps(res_json, cls=MyEncoder, ensure_ascii=False)

save_path = f'./output_json/v{version}'
if not os.path.exists(save_path):
    os.makedirs(save_path)

mpf_save_file = os.path.join(save_path, f'PerformanceEvaluation.json')
with open(mpf_save_file, 'w', encoding="utf8") as f:
    f.write(result_json)

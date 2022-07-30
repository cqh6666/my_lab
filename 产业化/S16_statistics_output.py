#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 09:18:38 2022

@author: liukang
"""
import json
import os

import numpy as np
import pandas as pd

test_data = pd.read_csv("output_json/input_csv/pred_raw_data/test_data_output.csv")
test_shap = pd.read_csv("output_json/input_csv/pred_raw_data/test_shap_output.csv")
train_data = pd.read_csv("output_json/input_csv/pred_raw_data/train_data_output.csv")
train_shap = pd.read_csv("output_json/input_csv/pred_raw_data/train_shap_output.csv")

first_X_feature_name = 'LIMIT_BAL'
last_X_feature_name = 'PAY_AMT6'
Label_name = 'default payment next month'
#shap_sum_name = 'shap_sum'
prediction_name = 'predict_prob'
personal_value_feature_name = 'BILL_AMT1'
group_num = 4
num_of_show_feature = 6

#ke hu qun ti hua xiang
#calculate data in total test set

#oupput_1: 总体违约率
avg_prediction = np.mean(test_data[prediction_name])

#output_2: 总体违约人数
expected_positive = int(test_data.shape[0] * avg_prediction)

person_expected_value = test_data[personal_value_feature_name] * test_data[prediction_name]

#output_3: 总体违约金额
total_expect_value = np.sum(person_expected_value.values)

# 最重要的6个特征
train_positive_sample_true = train_data.loc[:,Label_name] == 1
train_positive_shap_select = train_shap.loc[train_positive_sample_true]
train_negative_shap_select = train_shap.loc[~train_positive_sample_true]
mean_train_positive_shap = train_positive_shap_select.loc[:,first_X_feature_name:last_X_feature_name].mean(axis=0)
mean_train_negative_shap = train_negative_shap_select.loc[:,first_X_feature_name:last_X_feature_name].mean(axis=0)
InterCalss_ScoreDiff = mean_train_positive_shap - mean_train_negative_shap
InterCalss_ScoreDiff.sort_values(ascending=False,inplace=True)
important_feature = InterCalss_ScoreDiff.index.tolist()[:num_of_show_feature]

#calculate data in each group

#output_4: 客户群体画像
group_data_record = pd.DataFrame()

mean_feature_shap = test_shap.mean(axis=0)


"""
probability_range 风险范围
[0, 0.05]
[0.05 0.1]
[0.1 0.25]
[0.25 0.5]
[0.5 , ]

num 客户人数
share 客户占比
特征均值，特征shap均值，特征
"""
group_range_list = [
    [0, 0.05],
    [0.05, 0.1],
    [0.1, 0.25],
    [0.25, 0.5],
    [0.5, 1]
]
for i in range(len(group_range_list)):
    
    group_range = group_range_list[i]
    low_threshold = group_range[0]
    high_threshold = group_range[1]
    group_data_record.loc[i,'probability_range'] = '{}%-{}%'.format(int(low_threshold*100),int(high_threshold*100))
    sample_in_threshold_true = (test_data.loc[:,prediction_name] > low_threshold) & (test_data.loc[:,prediction_name] <= high_threshold)
    sample_in_threshold_select = test_data.loc[sample_in_threshold_true]
    # 每个群体的用户人数
    group_data_record.loc[i,'num'] = sample_in_threshold_select.shape[0]
    # 每个群体占总人数的占比
    group_data_record.loc[i,'share'] = sample_in_threshold_select.shape[0] / test_data.shape[0]
    
    sample_in_threshold_select_important_feature = sample_in_threshold_select.loc[:,important_feature]
    important_feature_mean = sample_in_threshold_select_important_feature.mean(axis=0)
    
    shap_in_threshold_select = test_shap.loc[sample_in_threshold_true]
    shap_in_threshold_select.reset_index(drop=True,inplace=True)
    shap_in_threshold_select_important_feature = shap_in_threshold_select.loc[:,important_feature]
    important_feature_shap_mean = shap_in_threshold_select_important_feature.mean(axis=0)
    
    group_sum_shape = np.log(shap_in_threshold_select.loc[:,prediction_name] / (1-shap_in_threshold_select.loc[:,prediction_name]))
    
    
    for j in range(num_of_show_feature):
        
        group_data_record.loc[i,'{}_mean'.format(important_feature[j])] = important_feature_mean[j]
        group_data_record.loc[i,'{}_shap_mean'.format(important_feature[j])] = important_feature_shap_mean[j]
        
        select_feature_mean_shap = mean_feature_shap[important_feature[j]]
        shap_in_threshold_diff_to_mean_shap = shap_in_threshold_select[important_feature[j]] - select_feature_mean_shap
        
        new_predict_score = group_sum_shape - shap_in_threshold_diff_to_mean_shap
        new_predict_proba = 1 / (1 + np.exp(-new_predict_score))
        mean_new_predict_proba = np.mean(new_predict_proba)
        mean_ori_predict_proba = np.mean(shap_in_threshold_select[prediction_name])
        mean_predict_proba_change = mean_ori_predict_proba - mean_new_predict_proba
        
        group_data_record.loc[i,'{}_proba_change_mean'.format(important_feature[j])] = '{}%'.format(round(mean_predict_proba_change*100,1))


def get_group_info(index):
    cur_group_info = group_data_record.iloc[index]
    probability_range = cur_group_info['probability_range']
    num = cur_group_info['num']
    share = cur_group_info['share']
    feature_importance_list = []
    for feature in important_feature:
        feature_importance_info = {
            "特征名称": feature,
            "特征值": cur_group_info[f'{feature}_mean'],
            "shap值": cur_group_info[f'{feature}_shap_mean'],
            "特征影响度": cur_group_info[f'{feature}_proba_change_mean']
        }
        feature_importance_list.append(feature_importance_info)
    return {
        "probability_range": probability_range,
        "num": num,
        "share": share,
        "important_feature": feature_importance_list
    }


# 保存为json格式
group_data_info = {
    "PremiumUserGroup": get_group_info(0),
    "RiskUserGroup": get_group_info(-1)
}

# save
result_json = json.dumps(group_data_info, ensure_ascii=False)
mpf_save_file = os.path.join(f'./output_json/v1/groupDataInfo.json')
with open(mpf_save_file, 'w', encoding="utf8") as f:
    f.write(result_json)


# 保存为csv
group_data_record.to_csv("./output_json/input_csv/analysis_data/statistics_output.csv")
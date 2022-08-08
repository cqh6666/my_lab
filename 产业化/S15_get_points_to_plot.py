# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     get_all_res_info
   Description:   ...
   Author:        cqh
   date:          2022/7/29 19:53
-------------------------------------------------
   Change Activity:
                  2022/7/29:
-------------------------------------------------
"""
__author__ = 'cqh'

import json
import os

from utils.data_utils import get_model_dict, get_train_test_X_y, NumpyEncoder
from utils.plot_utils import *
from utils.score_utils import get_all_info
import pandas as pd
import numpy as np
from utils.strategy_utils import train_strategy

"""
获取所有结果信息
"""
test_data = pd.read_csv("output_json/input_csv/pred_raw_data/test_data_output.csv")
test_shap = pd.read_csv("output_json/input_csv/pred_raw_data/test_shap_output.csv")
train_data = pd.read_csv("output_json/input_csv/pred_raw_data/train_data_output.csv")
train_shap = pd.read_csv("output_json/input_csv/pred_raw_data/train_shap_output.csv")
# 初始建模
my_best_model_select = ['best_xgb']
best_model_dict = get_model_dict(my_best_model_select, engineer=True)
clf = best_model_dict.get('best_xgb')

is_engineer = True
is_norm = True
train_data_x, test_data_x, train_data_y, test_data_y = get_train_test_X_y(is_engineer, is_norm)

train_prob, test_prob = train_strategy(clf, 3, train_data_x, train_data_y, test_data_x)

all_res = get_all_info(test_data_y, test_prob, train_prob)

# save
result_json = json.dumps(all_res, cls=NumpyEncoder)
version = 14
save_path = f'./output_json/v{version}'
if not os.path.exists(save_path):
    os.makedirs(save_path)

mpf_save_file = os.path.join(save_path, f'allValuePlotsInfo.json')
with open(mpf_save_file, 'w', encoding="utf8") as f:
    f.write(result_json)


# plot
curve_list = ['auroc','auprc','gini','ks','calibration','psi']
save_path = f'./output_json/v{version}/png'
if not os.path.exists(save_path):
    os.makedirs(save_path)
curve_file_list = [os.path.join(save_path, f"{curve_list[i]}.png") for i in range(len(curve_list))]
plot_auroc(curve_file_list[0], all_res['AUC']['plt_plot'][0], all_res['AUC']['plt_plot'][1])
# plot_auprc(curve_file_list[1], all_res['PRC']['plt_plot'][0], all_res['PRC']['plt_plot'][1])
plot_gini(curve_file_list[2], all_res['GINI']['plt_plot'][0], all_res['GINI']['plt_plot'][1])
plot_ks(curve_file_list[3], all_res['KS']['plt_plot'][0], all_res['KS']['plt_plot'][1])
plot_calibration_curve(curve_file_list[4], all_res['ECE']['plt_plot'][0], all_res['ECE']['plt_plot'][1])
# plot_psi(curve_file_list[5], all_res['PSI']['plt_plot'][0], all_res['PSI']['plt_plot'][1])
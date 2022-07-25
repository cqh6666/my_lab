# -*- coding: gbk -*-
"""
-------------------------------------------------
   File Name:     0007_get_shap
   Description:   ...
   Author:        cqh
   date:          2022/6/17 20:49
-------------------------------------------------
   Change Activity:
                  2022/6/17:
-------------------------------------------------
"""
__author__ = 'cqh'

import shap
import pandas as pd
import os
from xgb_utils_api import get_xgb_model_pkl
from utils_api import get_train_test_x_y


def get_shap_value(train_x, model):
    explainer = shap.TreeExplainer(model)
    shap_value = explainer.shap_values(train_x)
    res = pd.DataFrame(data=shap_value, columns=train_x.columns)
    res = res.abs().mean(axis=0)
    res = res / res.sum()
    res.fillna(0, inplace=True)
    return res


if __name__ == '__main__':
    pre_hour = 24

    XGB_MODEL_PATH = f"/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/code_pca/code_xgb/result/global_model/"

    glo_tl_boost_num = 500
    xgb_model_file = os.path.join(XGB_MODEL_PATH, f"0007_{pre_hour}h_global_xgb_boost{glo_tl_boost_num}.pkl")
    key_component = f"{pre_hour}_df_rm1_norm1"

    # get train data
    train_x, _, _, _ = get_train_test_x_y()
    # get xgb model
    xgb_model = get_xgb_model_pkl(is_transfer=1)

    # get shap value
    shap_weight = get_shap_value(train_x, xgb_model)
    shap_file_name = os.path.join(XGB_MODEL_PATH, f'0007_{pre_hour}h_global_xgb_shap_weight_boost500.csv')
    print(f"shap weight shape: {shap_weight.shape}")
    shap_weight.to_csv(shap_file_name, index=False)
    print(f"save success! - {shap_file_name}")
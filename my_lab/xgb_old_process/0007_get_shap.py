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
import pickle


def get_shap_value(train_x, model):
    explainer = shap.TreeExplainer(model)
    shap_value = explainer.shap_values(train_x)
    res = pd.DataFrame(data=shap_value, columns=train_x.columns)
    res = res.abs().mean(axis=0)
    res = get_normalize_weight(res)
    return res


def get_normalize_weight(weight):
    """return a pd.Series"""
    remained_feature_list = pd.read_csv(remained_feature_file, header=None).squeeze().tolist()
    result = pd.Series(index=remained_feature_list, dtype='float64')
    result.drop(['ID', 'Label'], axis=0, inplace=True)
    weight = pd.Series(weight, dtype='float64')
    result.loc[:] = weight
    result = result.abs() / result.abs().sum()
    result.fillna(0, inplace=True)
    return result


if __name__ == '__main__':
    pre_hour = 24
    root_dir = f"24h_old2"
    DATA_SOURCE_PATH = f"/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/data/{root_dir}/"
    XGB_MODEL_PATH = f'/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/result/psm_with_xgb/{root_dir}/global_model/'

    glo_tl_boost_num = 500
    remained_feature_file = os.path.join(DATA_SOURCE_PATH, f'remained_new_feature_map.csv')
    xgb_model_file = os.path.join(XGB_MODEL_PATH, f"0007_{pre_hour}h_global_xgb_boost{glo_tl_boost_num}.pkl")
    key_component = f"{pre_hour}_df_rm1_norm1"

    # get train data
    train_x = pd.read_feather(os.path.join(DATA_SOURCE_PATH, f"all_x_train_{key_component}.feather"))
    # get xgb model
    xgb_model = pickle.load(open(xgb_model_file, "rb"))

    # get shap value
    shap_weight = get_shap_value(train_x, xgb_model)
    shap_file_name = os.path.join(XGB_MODEL_PATH, f'0007_{pre_hour}h_shap_value_xgb_boost{glo_tl_boost_num}.csv')
    print(f"shap weight shape: {shap_weight.shape}")
    # save shap weight to csv
    shap_weight.to_csv(shap_file_name, index=True)
    print(f"save success! - {shap_file_name}")
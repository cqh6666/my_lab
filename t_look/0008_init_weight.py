"""
initial feature weight/metric for patient similarity
"""
import pickle

import pandas as pd
import shap


# --------------- pre para setting ---------------
# how many days in advance to predict
pre_day = 1
pre_hour = f'{pre_day * 24}h'
# the last folder of save file path
time_folder = f'{pre_day * 24}'
# data file path
data_folder = f'/panfs/pfs.local/work/liu/xzhang_sta/tangxizhuo/data/{time_folder}'
# pg model
pg_model = '/panfs/pfs.local/work/liu/xzhang_sta/tangxizhuo/pg'
# model file path
model_folder = f'{pg_model}/global_model'
# init metric folder
metric_folder = f'{pg_model}/init_metric'
# key name of data type
data_key = 'div2_snap1_rm2_miss4_norm2'
# remain feature
remain_key = 'snap1_rm2'
# init metric key name
importance_type = 'shap'
# global model para key
model_key = 'glo2'
# ------------------------------------------------


def get_shap_value(model):
    train_x = pd.read_feather(f'{data_folder}/{pre_hour}_train_x_{data_key}.feather')
    explainer = shap.TreeExplainer(model)
    shap_value = explainer.shap_values(train_x)
    res = pd.DataFrame(data=shap_value, columns=train_x.columns)
    res = res.abs().mean(axis=0)

    return res


def get_xgb_importance(model_path, kind):
    # ----- init feature weight according to global xgb model -----
    model = pickle.load(open(model_path, "rb"))
    # special case: shap value
    if kind == 'shap':
        weight_importance = get_shap_value(model)
    else:
        weight_importance = model.get_score(importance_type=kind)

    return weight_importance


def get_normalize_weight(weight):
    """return a pd.Series"""

    # all_feature: pd.Series, save all feature names
    all_feature = pd.read_csv(
        f'{data_folder}/{pre_hour}_{remain_key}_remain_feature.csv', header=None)[0]
    result = pd.Series(index=all_feature)
    # drop ID and Label
    result.drop(['ID', 'Label'], axis=0, inplace=True)
    # transform dict to pd.Series
    weight = pd.Series(weight)
    # len(result) usually > len(weight), extra values will be nan
    result.loc[:] = weight
    # normalize feature weight, sum(feature weight) = 1
    result = result.abs() / result.abs().sum()
    result.fillna(0, inplace=True)
    return result


def init_xgb_feature_metric():
    # load training data

    load_model_name = f'0006_{pre_hour}_xgb_{model_key}_{data_key}.pkl'
    load_model_path = f"{model_folder}/{load_model_name}"

    save_file_name = f'0008_{pre_hour}_xgb_{importance_type}_{model_key}_{data_key}.csv'
    save_file_path = f"{metric_folder}/{save_file_name}"

    # get feature weight(initial metric)
    init_metric = get_normalize_weight(get_xgb_importance(load_model_path, kind=importance_type))

    # save file
    init_metric.to_csv(save_file_path)


init_xgb_feature_metric()

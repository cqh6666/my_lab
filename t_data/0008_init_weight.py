"""
initial feature weight/metric for patient similarity
"""
import warnings
import pickle

import pandas as pd
import xgboost as xgb
from sklearn.metrics.pairwise import cosine_similarity
import shap

warnings.filterwarnings('ignore')


def get_xgb_para():
    """global xgb para"""
    params = {
        'booster': 'gbtree',
        'max_depth': 11,
        'min_child_weight': 7,
        'eta': 0.05,
        'objective': 'binary:logistic',
        'nthread': 20,
        'verbosity': 0,
        'eval_metric': 'logloss',
        'subsample': 1,
        'colsample_bytree': 0.7,
        'tree_method': 'hist',
        'seed': 1001,
    }
    num_boost_round = 500
    return params, num_boost_round


def get_shap_value(model, data_id):
    train_x = pd.read_feather(f'/panfs/pfs.local/work/liu/xzhang_sta/tangxizhuo/data/24/24h_train_x_{data_id}.feather')
    explainer = shap.TreeExplainer(model)
    shap_value = explainer.shap_values(train_x)
    res = pd.DataFrame(data=shap_value, columns=train_x.columns)
    res = res.abs().mean(axis=0)

    return res


def get_xgb_importance(model_name, kind, data_id):
    # ----- init feature weight according to global xgb model -----
    model = pickle.load(open(model_name, "rb"))
    # special case: shap value
    if kind == 'shap':
        weight_importance = get_shap_value(model, data_id)
    else:
        weight_importance = model.get_score(importance_type=kind)
    return weight_importance


def get_normalize_weight(weight):
    """return a pd.Series"""

    # all_feature: pd.Series, save all feature names
    all_feature = pd.read_csv(
        '/panfs/pfs.local/work/liu/xzhang_sta/tangxizhuo/data/24/24h_snap1_rm1_remain_feature.csv', header=None)[0]
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


def compare_cosine_distance():
    file_one_name = '24h_xgb_gain_para2_div1_snap1_rm1_miss2_norm1.csv'
    file_two_name = '24h_xgb_weight_para2_div1_snap1_rm1_miss2_norm1.csv'

    read_one = pd.read_csv(file_one_name, index_col=0).squeeze('columns').values
    read_two = pd.read_csv(file_two_name, index_col=0).squeeze('columns').values

    read_one = read_one.reshape((1, -1))
    read_two = read_two.reshape((1, -1))

    res = cosine_similarity(read_one, read_two)[0][0]

    # print the cos distance
    print("cos distance between two vector:", res)


def init_feature_metric():
    # load training data
    xgb_para_id = 'para2'
    data_id = 'div1_snap1_rm1_miss2_norm1'
    key_component_name = f'{xgb_para_id}_{data_id}'
    load_model_name = f'24h_xgb_glo_{key_component_name}.pkl'

    importance_type = 'shap'
    save_file_name = f'24h_xgb_{importance_type}_{key_component_name}.csv'

    # get feature weight(initial metric)
    init_metric = get_normalize_weight(get_xgb_importance(load_model_name, kind=importance_type, data_id=data_id))

    # save file
    init_metric.to_csv(save_file_name)


init_feature_metric()

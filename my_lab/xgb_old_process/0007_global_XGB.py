# encoding=gbk
"""
train a xgboost for data,
the parameters refer to BR

"""
import random

import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import pickle
import os
from my_logger import MyLog
import time
import sys
import shap


def get_xgb_params(num_boost):
    params = {
        'booster': 'gbtree',
        'max_depth': 11,
        'min_child_weight': 7,
        'subsample': 1,
        'colsample_bytree': 0.7,
        'eta': 0.05,
        'objective': 'binary:logistic',
        'nthread': 20,
        'verbosity': 0,
        'seed': 2022,
        'tree_method': 'hist'
    }
    num_boost_round = num_boost
    return params, num_boost_round


def xgb_train_global(train_x, train_y, num_boost, transfer_model=None, save=True):
    d_train = xgb.DMatrix(train_x, label=train_y)
    d_test = xgb.DMatrix(test_x, label=test_y)

    start = time.time()

    params, num_boost_round = get_xgb_params(num_boost)

    model = xgb.train(params=params,
                      dtrain=d_train,
                      num_boost_round=num_boost_round,
                      verbose_eval=False,
                      xgb_model=transfer_model)

    run_time = round(time.time() - start, 2)

    test_y_predict = model.predict(d_test)
    auc = roc_auc_score(test_y, test_y_predict)
    my_logger.info(f'train time: {run_time} | num_boost_round: {num_boost_round} : The auc of this model is {auc}')

    # save model
    if save:
        pickle.dump(model, open(xgb_global_model_file, "wb"))
        my_logger.warning(f"save xgb model to pkl - [{xgb_global_model_file}]")
        save_weight_importance(model)


def xgb_train_sub_global(num_boost, transfer_model=None, select_rate=0.1):
    train_x, train_y = get_sub_train_data(select_rate=select_rate)
    if transfer_model == None:
        transfer_flag = "transfer"
    else:
        transfer_flag = "no_transfer"
    my_logger.warning(f"local xgb train params: select_rate:{select_rate}, is_transfer:{transfer_flag}, num_boost:{num_boost}")
    xgb_train_global(train_x, train_y, num_boost=num_boost, transfer_model=transfer_model, save=False)


def save_weight_importance(model):
    # get weights of feature
    weight_importance = model.get_score(importance_type='weight')
    # gain_importance = model.get_score(importance_type='gain')
    # cover_importance = model.get_score(importance_type='cover')

    # 保存特征重要性
    importance_dict = {
        'weight': weight_importance,
        # 'gain': gain_importance,
        # 'cover': cover_importance
    }
    # 特征列表
    remained_feature_list = pd.read_csv(remained_feature_file, header=None).squeeze().tolist()

    for name, importance in importance_dict.items():
        result = pd.Series(index=remained_feature_list, dtype='float64')
        result.drop(['ID', 'Label'], axis=0, inplace=True)
        weight = pd.Series(importance, dtype='float64')
        result.loc[:] = weight
        result = result / result.sum()
        result.fillna(0, inplace=True)
        result.to_csv(init_psm_weight_file, index=False)
        my_logger.warning(f"save feature important weight to csv success! -{init_psm_weight_file}")


def get_important_weight(file_name):
    weight_result = os.path.join(MODEL_SAVE_PATH, file_name)
    normalize_weight = pd.read_csv(weight_result)
    my_logger.info(normalize_weight.shape)
    my_logger.info(normalize_weight.head())


def get_train_test_data(key_component):
    """
    获取就预处理方式的数据集：即 px,med换成7天内次数
    :return:
    """
    train_x = pd.read_feather(
        os.path.join(DATA_SOURCE_PATH, f"all_x_train_{key_component}.feather"))
    train_y = pd.read_feather(
        os.path.join(DATA_SOURCE_PATH, f"all_y_train_{key_component}.feather"))['Label']
    test_x = pd.read_feather(
        os.path.join(DATA_SOURCE_PATH, f"all_x_test_{key_component}.feather"))
    test_y = pd.read_feather(
        os.path.join(DATA_SOURCE_PATH, f"all_y_test_{key_component}.feather"))['Label']

    return train_x, train_y, test_x, test_y


def get_sub_train_data(select_rate):
    len_split = int(train_x.shape[0] * select_rate)
    random_idx = random.sample(list(range(train_x.shape[0])), len_split)

    train_x_ = train_x.loc[random_idx, :]
    train_x_.reset_index(drop=True, inplace=True)

    train_y_ = train_y.loc[random_idx]
    train_y_.reset_index(drop=True, inplace=True)

    return train_x_, train_y_


if __name__ == '__main__':
    # 自定义日志
    my_logger = MyLog().logger
    global_num_boost = 500
    local_num_boost = 50
    pre_hour = 24

    root_dir = f"{pre_hour}h_old2"
    DATA_SOURCE_PATH = f"/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/data/{root_dir}/"
    MODEL_SAVE_PATH = f'/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/result/psm_with_xgb/{root_dir}/global_model/'

    key_component = f"{pre_hour}_df_rm1_norm1"
    train_x, train_y, test_x, test_y = get_train_test_data(key_component)

    xgb_global_model_file = os.path.join(MODEL_SAVE_PATH, f'0007_{pre_hour}h_global_xgb_boost{global_num_boost}.pkl')
    init_psm_weight_file = os.path.join(MODEL_SAVE_PATH,
                                        f'0007_{pre_hour}h_global_xgb_feature_weight_boost{global_num_boost}.csv')
    remained_feature_file = os.path.join(DATA_SOURCE_PATH, f'remained_new_feature_map.csv')

    # global train
    xgb_train_global(train_x, train_y, num_boost=global_num_boost, save=True)

    # local train
    xgb_model = pickle.load(open(xgb_global_model_file, "rb"))
    # 迁移
    xgb_train_sub_global(num_boost=local_num_boost, transfer_model=xgb_model, select_rate=0.1)
    # 非迁移
    xgb_train_sub_global(num_boost=local_num_boost, select_rate=0.1)

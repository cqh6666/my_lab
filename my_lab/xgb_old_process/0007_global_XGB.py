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

def get_xgb_params():
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
        'seed': 998,
        'tree_method': 'hist'
    }
    num_boost_round = num_boost
    return params, num_boost_round


def xgb_train_global(train_x, train_y, save=False):
    d_train = xgb.DMatrix(train_x, label=train_y)
    d_test = xgb.DMatrix(test_x, label=test_y)

    params, num_boost_round = get_xgb_params()
    start = time.time()

    model = xgb.train(params=params,
                      dtrain=d_train,
                      num_boost_round=num_boost_round,
                      verbose_eval=False)
    run_time = round(time.time() - start, 2)

    test_y_predict = model.predict(d_test)
    auc = roc_auc_score(test_y, test_y_predict)
    my_logger.info(f'train time: {run_time} | num_boost_round: {num_boost_round} : The auc of this model is {auc}')

    # save model
    if save:
        model_file_name = os.path.join(MODEL_SAVE_PATH, f"{save_model_name}.pkl")
        pickle.dump(model, open(model_file_name, "wb"))
        my_logger.warning(f"save xgb model to pkl - [{model_file_name}]")
        save_weight_importance(model, num_boost_round)


def xgb_train_sub_global(select_rate=0.1):
    train_x, train_y = get_sub_train_data(select_rate=select_rate)
    my_logger.warning(f"start sub global({select_rate}) xgb training ...")
    xgb_train_global(train_x, train_y, False)

def save_weight_importance(model, num_boost_round):
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
        save_name = os.path.join(MODEL_SAVE_PATH, f'0007_{pre_hour}h_global_xgb_feature_{name}_boost{num_boost_round}.csv')
        result.to_csv(save_name, index=False)
        my_logger.warning(f"save feature important weight to csv success! -{save_name}")


def get_important_weight(boost, weight_name="weight"):
    file_name = f'0007_{pre_hour}h_global_xgb_feature_{weight_name}_boost{boost}.csv'
    weight_result = os.path.join(MODEL_SAVE_PATH, file_name)
    normalize_weight = pd.read_csv(weight_result)
    my_logger.info(normalize_weight.shape)
    my_logger.info(normalize_weight.head())


def get_train_test_data():
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
    num_boost = int(sys.argv[1])
    pre_hour = 24

    root_dir = f"{pre_hour}h_old2"
    DATA_SOURCE_PATH = f"/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/data/{root_dir}/"
    MODEL_SAVE_PATH = f'/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/result/psm_with_xgb/{root_dir}/global_model/'

    key_component = f"{pre_hour}_df_rm1_norm1"

    train_x, train_y, test_x, test_y = get_train_test_data()

    save_model_name = f'0007_{pre_hour}h_global_xgb_boost{num_boost}'
    remained_feature_file = os.path.join(DATA_SOURCE_PATH, f'remained_new_feature_map.csv')

    # xgb_train_global(train_x, train_y, save=True)
    xgb_train_sub_global(select_rate=0.1)

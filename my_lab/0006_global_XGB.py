# encoding=gbk
"""
train a xgboost for data,
the parameters refer to BR

"""
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import warnings
import time
import pickle
import os
from my_logger import MyLog
import sys

# 自定义日志
my_logger = MyLog().logger

warnings.filterwarnings('ignore')

num_boost = 1
key_component = '24h_all_999_normalize'
train_x = pd.read_feather(
    f'/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/data/24h/{key_component}_train_x_data.feather')
train_y = \
    pd.read_feather(f'/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/data/24h/{key_component}_train_y_data.feather')[
        'Label']

test_x = pd.read_feather(
    f'/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/data/24h/{key_component}_test_x_data.feather')
test_y = \
    pd.read_feather(f'/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/data/24h/{key_component}_test_y_data.feather')[
        'Label']

SAVE_PATH = '/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/result/personal_model_with_xgb/'
save_model_name = f'0006_xgb_global_{key_component}_{num_boost}.pkl'
remain_feature_file = '/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/data/24h/24h_999_remained_feature.csv'


def xgb_train_global(num_boost_round):
    d_train = xgb.DMatrix(train_x, label=train_y)
    d_test = xgb.DMatrix(test_x, label=test_y)

    params = {
        'booster': 'gbtree',
        'max_depth': 8,
        'min_child_weight': 10,
        'subsample': 0.5,
        'colsample_bytree': 0.5,
        'eta': 0.15,
        'objective': 'binary:logistic',
        'nthread': 20,
        'verbosity': 0,
        'eval_metric': 'logloss',
        'seed': 999,
        'tree_method': 'hist',
        "early_stopping_rounds": num_boost_round / 2
    }

    # cv_result = xgb.cv(params, d_train, num_boost_round=500, early_stopping_rounds=100, nfold=5, metrics=['auc'])

    evals_result = {}
    model = xgb.train(params=params,
                      dtrain=d_train,
                      evals=[(d_test, 'test')],
                      num_boost_round=num_boost_round,
                      evals_result=evals_result,
                      verbose_eval=False)
    # print(evals_result)
    test_y_predict = model.predict(d_test)
    auc = roc_auc_score(test_y, test_y_predict)
    my_logger.info(f'num_boost_round: {num_boost_round} : The auc of this model is {auc}.\n')

    # save model
    pickle.dump(model, open(os.path.join(SAVE_PATH, save_model_name), "wb"))

    # get weights of feature
    weight_importance = model.get_score(importance_type='weight')
    gain_importance = model.get_score(importance_type='gain')
    cover_importance = model.get_score(importance_type='cover')

    # 保存特征重要性
    importance_dict = {
        'weight_importance': weight_importance,
        'gain_importance': gain_importance,
        'cover_importance': cover_importance
    }
    all_feature = pd.read_csv(remain_feature_file, header=None)[0]
    for name, importance in importance_dict.items():
        result = pd.Series(index=all_feature)
        result.drop(['ID', 'Label'], axis=0, inplace=True)
        weight = pd.Series(importance)
        result.loc[:] = weight
        result = result / result.sum()
        result.fillna(0, inplace=True)
        save_name = f'0006_xgb_global_feature_{name}_boost{num_boost_round}_v0.csv'
        result.to_csv(os.path.join(SAVE_PATH, save_name), index=False)

    my_logger.info(f'num_boost_round: {num_boost_round} : save with csv success!')


def get_normalize_weight():
    file_name = '0006_xgb_global_feature_weight_importance_boost91_v0.csv'
    file = os.path.join(SAVE_PATH, file_name)
    normalize_weight = pd.read_csv(file)
    my_logger.info(normalize_weight.shape)
    my_logger.info(normalize_weight.head())



if __name__ == '__main__':
    # boost_nums = int(sys.argv[1])
    # xgb_train_global(boost_nums)
    get_normalize_weight()
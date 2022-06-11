# encoding=gbk
"""
train a xgboost for data,
the parameters refer to BR

"""
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import pickle
import os
from my_logger import MyLog
import time
import sys

def xgb_train_global(num_boost_round):
    d_train = xgb.DMatrix(train_x, label=train_y)
    d_test = xgb.DMatrix(test_x, label=test_y)

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
    model_file_name = os.path.join(MODEL_SAVE_PATH, save_model_name)
    pickle.dump(model, open(model_file_name, "wb"))
    my_logger.info(f"save xgb model to pkl - [{model_file_name}]")

    # get weights of feature
    weight_importance = model.get_score(importance_type='weight')
    gain_importance = model.get_score(importance_type='gain')
    cover_importance = model.get_score(importance_type='cover')

    # 保存特征重要性
    importance_dict = {
        'weight': weight_importance,
        'gain': gain_importance,
        'cover': cover_importance
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
        save_name = f'0006_xgb_global_feature_{name}_boost{num_boost_round}.csv'
        result.to_csv(os.path.join(MODEL_SAVE_PATH, save_name), index=False)

    my_logger.info(f'num_boost_round: {num_boost_round} | save with csv success!')


def get_normalize_weight(boost, weight_name="weight"):
    file_name = f'0006_xgb_global_feature_{weight_name}_boost{boost}.csv'
    weight_result = os.path.join(MODEL_SAVE_PATH, file_name)
    normalize_weight = pd.read_csv(weight_result)
    my_logger.info(normalize_weight.shape)
    my_logger.info(normalize_weight.head())


if __name__ == '__main__':
    # 自定义日志
    my_logger = MyLog().logger

    num_boost = int(sys.argv[1])
    pre_hour = 24
    key_component = '24h_all_999_norm_miss'

    DATA_SOURCE_PATH = f"/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/data/{pre_hour}h/"
    MODEL_SAVE_PATH = f'/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/result/personal_model_with_xgb/{pre_hour}h_xgb_model'
    FEATURE_MAP_PATH = "/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/data/"

    train_x = pd.read_feather(os.path.join(DATA_SOURCE_PATH, f"all_x_train_{pre_hour}h_norm_dataframe_999_miss_medpx_max2dist.feather"))
    train_y = pd.read_feather(os.path.join(DATA_SOURCE_PATH, f"all_y_train_{pre_hour}h_norm_dataframe_999_miss_medpx_max2dist.feather"))['Label']
    test_x = pd.read_feather(os.path.join(DATA_SOURCE_PATH, f"all_x_test_{pre_hour}h_norm_dataframe_999_miss_medpx_max2dist.feather"))
    test_y = pd.read_feather(os.path.join(DATA_SOURCE_PATH, f"all_y_test_{pre_hour}h_norm_dataframe_999_miss_medpx_max2dist.feather"))['Label']

    save_model_name = f'0006_xgb_global_{key_component}_boost{num_boost}.pkl'
    remained_feature_file = os.path.join(FEATURE_MAP_PATH, f'{pre_hour}_999_remained_new_feature_map.csv')

    xgb_train_global(num_boost)
    # get_normalize_weight(num_boost, "weight")

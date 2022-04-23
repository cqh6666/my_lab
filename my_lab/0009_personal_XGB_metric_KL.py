#encoding=gbk
"""
build personal xgb models for test data with learned KL metric
"""
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
import threading
import pandas as pd
import xgboost as xgb
from gc import collect
import warnings
import sys
import time
from multiprocessing import cpu_count
from my_logger import MyLog
from memory_profiler import profile

warnings.filterwarnings('ignore')
profile_log = open('/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/code/log/profile_log.log','w+')
my_logger = MyLog().logger

my_logger.info(f"cpu_count: {cpu_count()}")


def get_global_xgb_para():
    """global xgb para"""
    params = {
        'booster': 'gbtree',
        'max_depth': 8,
        'min_child_weight': 7,
        'eta': 0.15,
        'objective': 'binary:logistic',
        'nthread': 20,
        'verbosity': 0,
        'eval_metric': 'logloss',
        'subsample': 0.5,
        'colsample_bytree': 0.5,
        'tree_method': 'hist',
        'seed': 1001,
    }
    num_boost_round = 300
    return params, num_boost_round


def get_local_xgb_para():
    """personal xgb para"""
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
        'seed': 1001,
        'tree_method': 'hist'
    }
    num_boost_round = 1
    return params, num_boost_round


def get_global_xgb():
    """train a global xgb model"""
    params, num_boost_round = get_global_xgb_para()
    d_train_global = xgb.DMatrix(data=train_x, label=train_y)
    model = xgb.train(params=params,
                      dtrain=d_train_global,
                      num_boost_round=num_boost_round,
                      verbose_eval=False)
    return model


@profile(precision=4, stream=profile_log)
def personalized_modeling(pre_data, idx, x_test):
    """build personal model for target sample"""

    personalized_modeling_start_time = time.time()
    similar_rank = pd.DataFrame()

    similar_rank['data_id'] = train_x.index.tolist()
    similar_rank['Distance'] = (abs((train_x - pre_data) * feature_weight)).sum(axis=1)

    similar_rank.sort_values('Distance', inplace=True)
    similar_rank.reset_index(drop=True, inplace=True)
    select_id = similar_rank.iloc[:len_split, 0].values

    select_train_x = train_x.iloc[select_id, :]
    select_train_y = train_y.iloc[select_id]
    fit_train = select_train_x
    fit_test = x_test

    sample_ki = similar_rank.iloc[:len_split, 1].tolist()
    sample_ki = [(sample_ki[0] + m_sample_weight) / (val + m_sample_weight) for val in sample_ki]

    d_train_local = xgb.DMatrix(fit_train, label=select_train_y, weight=sample_ki)
    params, num_boost_round = get_local_xgb_para()

    # use transform
    xgb_local = xgb.train(params=params,
                          dtrain=d_train_local,
                          num_boost_round=num_boost_round,
                          xgb_model=xgb_global,
                          verbose_eval=False)

    # # not use transform
    # xgb_local = xgb.train(params=params,
    #                       dtrain=d_train_local,
    #                       num_boost_round=num_boost_round,
    #                       verbose_eval=False)

    d_test_local = xgb.DMatrix(fit_test)
    proba = xgb_local.predict(d_test_local)

    global_lock.acquire()
    test_result.loc[idx, 'proba'] = proba
    p_weight.loc[idx, :] = xgb_local.get_score(importance_type='weight')
    global_lock.release()

    my_logger.info(f"{idx} - time: {time.time() - personalized_modeling_start_time}")


# ----- work space -----
# read data
train_x = pd.read_feather('/panfs/pfs.local/work/liu/xzhang_sta/tangxizhuo/data/24'
                          '/24h_train_x_div1_snap1_rm1_miss1_norm1.feather')
train_y = pd.read_feather('/panfs/pfs.local/work/liu/xzhang_sta/tangxizhuo/data/24'
                          '/24h_train_y_div1_snap1_rm1_miss1_norm1.feather')['Label']
test_x = pd.read_feather('/panfs/pfs.local/work/liu/xzhang_sta/tangxizhuo/data/24'
                         '/24h_test_x_div1_snap1_rm1_miss1_norm1.feather')
test_y = pd.read_feather('/panfs/pfs.local/work/liu/xzhang_sta/tangxizhuo/data/24'
                         '/24h_test_y_div1_snap1_rm1_miss1_norm1.feather')['Label']

# read learned KL metric
# learned_metric_iteration = 120
learned_metric_iteration = str(sys.argv[3])
# 读取迭代了k次的特征权重csv文件
feature_weight = pd.read_csv(f'/panfs/pfs.local/work/liu/xzhang_sta/tangxizhuo/pg/KL_result/0008_01_001_005-1_{learned_metric_iteration}.csv')
feature_weight = feature_weight.iloc[:, 0].tolist()

# personal para setting
select_ratio = 0.1
m_sample_weight = 0.01

n_thread = 20
# range of test
# start_idx = 0
start_idx = int(sys.argv[1])
final_idx = test_x.shape[0]
# end_idx = 20
end_idx = int(sys.argv[2])
end_idx = final_idx if end_idx > final_idx else end_idx
my_logger.info(f"the idx range is: [{start_idx},{end_idx}]")

# the number of selected train data
len_split = int(train_x.shape[0] * select_ratio)

# init test result
test_result = pd.DataFrame(columns=['real', 'proba'])
test_result['real'] = test_y.iloc[start_idx:end_idx]

# init p_weight to save weight importance for each personalized model
p_weight = pd.DataFrame(index=test_result.index.tolist(), columns=test_x.columns.tolist())

# init thread list
thread_list = []
# init thread pool
pool = ThreadPoolExecutor(max_workers=n_thread)
# get thread lock
global_lock = threading.Lock()

# get global xgb model
xgb_global = get_global_xgb()

# build personalized model for each test sample
for test_idx in range(start_idx, end_idx):
    pre_data_select = test_x.loc[test_idx, :]
    x_test_select = test_x.loc[[test_idx], :]

    # execute multi threads
    thread = pool.submit(personalized_modeling, pre_data_select, test_idx, x_test_select)
    thread_list.append(thread)
    collect()

# wait for all threads completing
wait(thread_list, return_when=ALL_COMPLETED)

# ----- save result -----
# xxx_{select ratio}_{sample weight}_{regularization}-{KL ID}_{start id}_{whether transfrom}
test_result.to_csv(f'/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/result/personal_model_with_xgb/test_result/0009_{learned_metric_iteration}_proba_tran_{start_idx}_{end_idx}.csv', index=True)

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 12:07:48 2018

@author: Shuxy
"""
# import multiprocessing
import time
import sys
import threading
import numpy as np
import pandas as pd
from random import shuffle
from sklearn.linear_model import LogisticRegression, LinearRegression
from gc import collect

import warnings

# warnings.filterwarnings('ignore')


# read data, check dtpye
train_df = pd.read_csv('/panfs/pfs.local/work/liu/xzhang_sta/yuanborong/data/df_data/pre_48h/train-data-no-empty.csv',
                       dtype=np.float32)
# test_df = pd.read_csv('/panfs/pfs.local/work/liu/xzhang_sta/yuanborong/data/df_data/pre_48h/test-data-no-empty.csv',dtype=np.float32)
# over

# feature_max = train_df.loc[:,:'MED_9995'].max(axis=0)
# train_df.loc[:,:'MED_9995'] = train_df.loc[:,:'MED_9995'] / feature_max
feature_happened = train_df.loc[:, :'MED_9995'] > 0
feature_happened_count = feature_happened.sum(axis=0)
feature_sum = train_df.loc[:, :'MED_9995'].sum(axis=0)
feature_average_if = feature_sum / feature_happened_count
train_df.loc[:, :'MED_9995'] = train_df.loc[:, :'MED_9995'] / feature_average_if

del feature_happened

# learning_rate = 0.1
ki_number = int(sys.argv[1])
step = 5
l_rate = 0.00001
Iteration = ki_number + 1
Number_of_Iteration = 1000
threading_round = 50
threading_num = 20
k = 0.05
# regularization parameters
regularization_c = 0.05
m_sample_ki = 0.01
global_lock = threading.Lock()
# train_original save ori_auc
# train_original = train_df.copy()
# test_original = test_df.copy()

lr_All = LogisticRegression(solver='liblinear')
X_train = train_df.drop(['Label'], axis=1)
y_train = train_df['Label']
# X_test = test_df.drop(['Label'], axis=1)

lr_All.fit(X_train, y_train)
# test_original['predict_proba'] = lr_All.predict_proba(X_test)[:, 1]

X_columns_name = X_train.columns
del X_train
del y_train
# del X_test

# 通过LR学习到的权重向量
Weight_importance = lr_All.coef_[0]
# clean_idx = 0


# for update weight_importance
if ki_number == 0:

    ki = [abs(i) for i in Weight_importance]
    ki = [i / sum(ki) for i in ki]

else:
    ki = pd.read_csv(
        '/panfs/pfs.local/work/liu/xzhang_sta/yuanborong/kang_test/result/Ma_old_Nor_Gra_mp_ver2_01_001_0_005-1_{}.csv'.format(
            ki_number))
    ki = ki['Ma_update_{}'.format(ki_number)].tolist()


# print('sum_ki_{}'.format(np.sum(ki)))

def learn_similarity_measure(pre_data, true, I_idx, X_test):
    # mat_copy = train_rank_X - pre_data
    # mat_copy *= ki

    # mat_copy = abs(mat_copy)

    similar_rank = pd.DataFrame()

    similar_rank['data_id'] = train_rank.index.tolist()
    # 计算距离
    similar_rank['Distance'] = (abs((train_rank_X - pre_data) * ki)).sum(axis=1)

    # similar_rank['Distance'] = mat_copy.sum(axis=1)
    # mat_copy = []

    similar_rank.sort_values('Distance', inplace=True)
    similar_rank.reset_index(drop=True, inplace=True)
    select_id = similar_rank.iloc[:len_split, 0].values

    # print(np.sum(train_rank['Demo1']))

    train_data = train_rank.iloc[select_id, :]
    X_train = train_data.loc[:, :'MED_9995']
    fit_train = X_train * Weight_importance
    y_train = train_data['Label']

    # print('shape_{}'.format(train_data.shape[0]))
    # print(np.mean(similar_rank.iloc[:len_split, 1].values))
    # print(similar_rank.iloc[0, 1])

    fit_test = X_test * Weight_importance

    sample_ki = similar_rank.iloc[:len_split, 1].tolist()
    sample_ki = [(sample_ki[0] + m_sample_ki) / (val + m_sample_ki) for val in sample_ki]

    # 加权后再训练
    lr = LogisticRegression(solver='liblinear')
    lr.fit(fit_train, y_train, sample_ki)

    proba = lr.predict_proba(fit_test)[:, 1]

    X_train = X_train - pre_data
    X_train = abs(X_train)
    mean_r = np.mean(X_train)
    y = abs(true - proba)

    global Iteration_data
    global y_Iteration

    global_lock.acquire()

    Iteration_data.loc[I_idx, :] = mean_r
    y_Iteration[I_idx] = y

    global_lock.release()

    # print('sum_meanr_{}'.format(np.sum(mean_r)))
    # print('y_{}'.format(y))


# min weight of sample for logisticregression
for k_idx in range(Iteration, Iteration + step):
    last_idx = list(range(len(train_df)))
    shuffle(last_idx)
    last_data = train_df
    last_data = last_data.loc[last_idx, :]
    last_data.reset_index(drop=True, inplace=True)
    # last_data['predict_proba'] = lr_All.predict_proba(last_data.drop('Label', axis=1))[:, 1]

    Iteration_data = pd.DataFrame(index=range(Number_of_Iteration), columns=X_columns_name)
    y_Iteration = pd.Series(index=range(Number_of_Iteration))
    I_idx_now = 0

    select_data = last_data.loc[:Number_of_Iteration - 1, :]
    select_data.reset_index(drop=True, inplace=True)

    train_rank = last_data.loc[Number_of_Iteration - 1:, :].copy()
    train_rank.reset_index(drop=True, inplace=True)
    train_rank_X = train_rank.drop('Label', axis=1)
    len_split = int(len(train_rank) * k)
    train_rank_dtype = train_rank.dtypes
    # ues_core = int(multiprocessing.cpu_count())
    # pool = multiprocessing.Pool(processes=ues_core)

    for threading_round_idx in range(threading_round):

        threadList = []

        for threading_num_idx in range(threading_num):
            s_idx = (threading_round_idx * threading_num) + threading_num_idx

            pre_data_select = select_data.loc[s_idx, :'MED_9995']
            true_select = select_data.loc[s_idx, 'Label']
            X_test_select = select_data.loc[[s_idx], :'MED_9995']

            thread = threading.Thread(target=learn_similarity_measure,
                                      args=(pre_data_select, true_select, I_idx_now, X_test_select))
            thread.start()
            threadList.append(thread)
            I_idx_now += 1

        for join_num in range(threading_num):
            threadList[join_num].join()

            # pool.apply_async(learn_similarity_measure, args=(train_rank_data,pre_data_select,true_select,I_idx_now,X_test_select,))

        # clean_idx += 1
        # if clean_idx % 20 == 0:
        collect()

        # else:
        # continue

    # pool.close()
    # pool.join()
    new_similar = Iteration_data * ki
    y_pred = new_similar.sum(axis=1)

    new_ki = []
    risk_gap = [real - pred for real, pred in zip(list(y_Iteration), list(y_pred))]
    for idx, value in enumerate(ki):
        features_x = list(Iteration_data.iloc[:, idx])
        plus_list = [a * b for a, b in zip(risk_gap, features_x)]
        new_value = value + l_rate * (sum(plus_list) - regularization_c * value)
        # print('new_value_{}'.format(new_value))
        new_ki.append(new_value)

    new_ki = list(map(lambda x: x if x > 0 else 0, new_ki))
    ki = new_ki.copy()

    if k_idx % step == 0:
        table = pd.DataFrame({'Ma_update_{}'.format(k_idx): ki})
        print(time.time())
        table.to_csv(
            '/panfs/pfs.local/work/liu/xzhang_sta/yuanborong/kang_test/result/Ma_old_Nor_Gra_mp_ver2_005_001_0_005-1_{}.csv'.format(
                k_idx), index=False)
        print(time.time())

    else:
        continue

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 12:07:48 2018

@author: Shuxy
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score
from gc import collect
from time import sleep
import os
import warnings
warnings.filterwarnings('ignore')


#read data
train_df = pd.read_csv('/panfs/pfs.local/work/liu/xzhang_sta/yuanborong/data/df_data/pre_48h/train-data-no-empty.csv',dtype=np.float32)
test_df = pd.read_csv('/panfs/pfs.local/work/liu/xzhang_sta/yuanborong/data/df_data/pre_48h/test-data-no-empty.csv',dtype=np.float32)
#over

feature_happened = train_df.loc[:,:'MED_9995'] > 0
feature_happened_count = feature_happened.sum(axis=0)
feature_sum = train_df.loc[:,:'MED_9995'].sum(axis=0)
feature_average_if = feature_sum / feature_happened_count
train_df.loc[:,:'MED_9995'] = train_df.loc[:,:'MED_9995'] / feature_average_if
test_df.loc[:,:'MED_9995'] = test_df.loc[:,:'MED_9995'] / feature_average_if
del feature_happened

#train_original = train_df.copy()
test_original = test_df.copy()

lr_All = LogisticRegression(solver='liblinear', class_weight='balanced', max_iter=200)
global_X_train = train_df.drop(['Label'], axis=1)
y_train = train_df['Label']
X_test = test_df.drop(['Label'], axis=1)

lr_All.fit(global_X_train, y_train)
test_original['predict_proba'] = lr_All.predict_proba(X_test)[:, 1]
    
print(roc_auc_score(test_original['Label'],test_original['predict_proba']))


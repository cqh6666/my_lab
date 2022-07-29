# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     testss
   Description:   ...
   Author:        cqh
   date:          2022/7/27 14:13
-------------------------------------------------
   Change Activity:
                  2022/7/27:
-------------------------------------------------
"""
__author__ = 'cqh'
# 原始数据
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import brier_score_loss

from utils.data_utils import get_model_dict
from utils.strategy_utils import smote_process

all_data_x, all_data_y = get_data_X_y()
# SMOTE处理
all_data_x, all_data_y = smote_process(all_data_x, all_data_y)

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(all_data_x, all_data_y, test_size=0.3,
                                                        random_state=2022)

model = get_model_dict(['lr']).get('lr')
model.fit(X_train, y_train)
train_prob = model.predict_proba(X_train)[:, 1]
test_prob = model.predict_proba(X_test)[:, 1]
clf = CalibratedClassifierCV(model, cv="prefit")
clf.fit(X_train, y_train)
train_prob2 = model.predict_proba(X_train)[:, 1]
test_prob2 = model.predict_proba(X_test)[:, 1]

observation, prediction = calibration_curve(y_test, test_prob2, n_bins=10, strategy='quantile')
brier_score = np.mean(np.square(np.array(observation - prediction)))
brier_score2 = brier_score_loss(y_test, test_prob)
brier_score3 = brier_score_loss(y_test, test_prob2)
print(brier_score, brier_score2, brier_score3)


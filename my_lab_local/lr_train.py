# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     lr_train
   Description:   多进程跑全局LR
   Author:        cqh
   date:          2022/5/23 9:23
-------------------------------------------------
   Change Activity:
                  2022/5/23:
-------------------------------------------------
"""
__author__ = 'cqh'

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from my_logger import MyLog
from sklearn.metrics import roc_auc_score
import pandas as pd
import os

def train_logistic_regression(train_x, train_y, test_x, test_y, max_iter):
    clf = LogisticRegression(solver='liblinear', max_iter=max_iter)
    clf.fit(train_x, train_y)

    # feature weight
    weight_importance = clf.coef_[0]
    weight_importance = [abs(i) for i in weight_importance]
    weight_importance = [i / sum(weight_importance) for i in weight_importance]
    weight_importance_df = pd.DataFrame({"init_weight": weight_importance})
    weight_importance_df.to_csv(r"D:\lab\other_file\24h_global_lr.csv", index=False)

    # predict_y = clf.predict_proba(test_x)[:, 1]
    # score = roc_auc_score(test_y, predict_y)
    score = roc_auc_score(test_y, clf.decision_function(test_x))
    print(score)

if __name__ == '__main__':
    breast_cancer = load_breast_cancer()
    data_x = breast_cancer.data
    data_y = breast_cancer.target

    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=45)

    train_logistic_regression(x_train, y_train, x_test, y_test, 1000)



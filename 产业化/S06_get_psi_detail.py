# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     S06_get_psi_detail
   Description:   ...
   Author:        cqh
   date:          2022/7/26 19:46
-------------------------------------------------
   Change Activity:
                  2022/7/26:
-------------------------------------------------
"""
__author__ = 'cqh'

from sklearn.datasets import load_iris
import numpy as np
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)

random_state = np.random.RandomState(0)
n_samples, n_features = X.shape
X = np.concatenate([X, random_state.randn(n_samples, 200 * n_features)], axis=1)

# Limit to the two first classes, and split into training and test
X_train, X_test, y_train, y_test = train_test_split(
    X[y < 2], y[y < 2], test_size=0.5, random_state=random_state
)

# logistic regression ------
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(penalty='l1', C=0.9, solver='saga')
lr.fit(X_train, y_train)

# predicted proability
pred_train = lr.predict_proba(X_train)[:, 1]
pred_test = lr.predict_proba(X_test)[:, 1]

# 调用scorecardpy
# ks = sc.perf_eva(y_train, pred_train, title="train", show_plot=False)
# roc = sc.perf_eva(y_test, pred_test, title="test", show_plot=False)
# psi = sc.perf_psi(
#     {'train':pd.DataFrame(pred_train), 'test':pd.DataFrame(pred_test)},
#     label = {'train':y_train, 'test':y_test},
# )

from toad.metrics import PSI

# ks2 = KS(pred_test, y_test)
psi2 = PSI(pred_test, pred_train)

from utils.score_utils import get_psi

psi3 = get_psi(pred_test, pred_train)
# ks3 = get_ks(y_test, pred_test)

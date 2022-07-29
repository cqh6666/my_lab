# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from bayes_opt import BayesianOptimization

from utils.data_utils import get_train_test_X_y

is_engineer = False
is_norm = True
train_x, test_x, train_y, test_y = get_train_test_X_y(is_engineer, is_norm)
print('train x shape:', train_x.shape, 'test x shape:', test_x.shape)
print("is_engineer", is_engineer, "is_norm", is_norm)

all_data_x = pd.concat([train_x, test_x], axis=0)
all_data_y = pd.concat([train_y, test_y], axis=0)


def RF_evaluate(n_estimators, min_samples_split, max_features, max_depth):
    val = cross_val_score(
        RandomForestClassifier(n_estimators=int(n_estimators),
                               min_samples_split=int(min_samples_split),
                               max_features=min(max_features, 0.999),
                               max_depth=int(max_depth),
                               random_state=2022,
                               n_jobs=-1),
        all_data_x, all_data_y, scoring='roc_auc', cv=5
    ).mean()

    return val


# 确定取值空间
pbounds = {'n_estimators': (10, 250),  # 表示取值范围为10至250
           'min_samples_split': (2, 25),
           'max_features': (0.1, 0.999),
           'max_depth': (5, 12)}

RF_bo = BayesianOptimization(
    f=RF_evaluate,  # 目标函数
    pbounds=pbounds,  # 取值空间
    verbose=2,  # verbose = 2 时打印全部，verbose = 1 时打印运行中发现的最大值，verbose = 0 将什么都不打印
    random_state=1,
)

RF_bo.maximize(init_points=10,  # 随机搜索的步数
               n_iter=1000,  # 执行贝叶斯优化迭代次数
               acq='ei')

print(RF_bo.max)
res = RF_bo.max
params_max = res['params']
print(params_max)
params_max['n_estimators'] = int(params_max['n_estimators'])
params_max['min_samples_split'] = int(params_max['min_samples_split'])
params_max['max_features'] = min(params_max['max_features'], 0.999)
params_max['max_depth'] = int(params_max['max_depth'])
print(params_max)

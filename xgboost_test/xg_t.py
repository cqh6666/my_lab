# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     xg_t
   Description:   ...
   Author:        cqh
   date:          2022/4/15 11:14
-------------------------------------------------
   Change Activity:
                  2022/4/15:
-------------------------------------------------
"""
__author__ = 'cqh'

from sklearn.datasets import load_iris
import xgboost as xgb
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
from memory_profiler import profile

profiler_log = open('./log/memory_profiler.log', 'w+')


@profile(precision=4, stream=profiler_log)
def run():
    iris = load_iris()

    x = iris.data
    y = iris.target

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=45)

    params = {
        'booster': 'gbtree',
        'objective': 'multi:softmax',  # 回归任务设置为：'objective': 'reg:gamma',
        'num_class': 3,  # 回归任务没有这个参数
        'gamma': 0.1,
        'max_depth': 6,
        'lambda': 2,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'min_child_weight': 3,
        'silent': 1,
        'eta': 0.1,
        'seed': 1000,
        'nthread': 4,
    }

    d_train = xgb.DMatrix(x_train, y_train)
    model = xgb.train(params, d_train, 500)

    d_test = xgb.DMatrix(x_test)
    ans = model.predict(d_test)

    # 计算准确率
    cnt1 = 0
    cnt2 = 0
    for i in range(len(y_test)):
        if ans[i] == y_test[i]:
            cnt1 += 1
        else:
            cnt2 += 1

    print("Accuracy: %.2f %% " % (100 * cnt1 / (cnt1 + cnt2)))

    # 显示重要特征
    plot_importance(model)


if __name__ == '__main__':
    for i in range(5):
        run()


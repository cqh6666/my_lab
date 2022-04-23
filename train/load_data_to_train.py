# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     load_data_to_train
   Description :
   Author :       cqh
   date：          2022/4/13 15:50
-------------------------------------------------
   Change Activity:
                   2022/4/13:
-------------------------------------------------
"""
__author__ = 'cqh'

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import precision_score, recall_score, roc_auc_score
import xgboost as xgb
import numpy as np
from matplotlib import pyplot as plt
import logging

source_path = r"D:\\lab\\feather\\iris_data.feather"

# data = pd.read_feather(source_path)
breast_cancer = load_breast_cancer()
data_x = breast_cancer.data
data_y = breast_cancer.target
# data_x = data.iloc[:, :-1]
# data_y = data.iloc[:, -1]


x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.25)

# x_train.reset_index(drop=True, inplace=True)
# x_train.to_feather('../feather/x_train.feather')

# 转换为DMatrix数据格式
dtrain = xgb.DMatrix(x_train, label=y_train)
dtest = xgb.DMatrix(x_test, label=y_test)

# 设置参数
parameters = {
    'n_estimators': 5,
    'eta': 0.3,
    'objective': 'binary:logistic',  # error evaluation for multiclass tasks
    'num_class': 3,  # number of classes to predic
    'max_depth': 3  # depth of the trees in the boosting process
}
num_round = 100  # the number of training iterations

# 模型训练
bst = xgb.train(parameters, dtrain, num_boost_round=num_round)

# 输出树结构
tree_df = bst.trees_to_dataframe()
print(tree_df[tree_df['Tree'] == 1])
# xgb.plotting.plot_tree(bst, num_trees=0)

# 模型预测
preds = bst.predict(dtest)
print(preds)
print("========================")
weight_1 = bst.get_fscore()
weight_2 = bst.get_score(importance_type='weight')
weight_3 = bst.get_score(importance_type='gain')
weight_4 = bst.get_score(importance_type='cover')
weight_5 = bst.get_score(importance_type='total_gain')
weight_6 = bst.get_score(importance_type='total_cover')
print("weight_1", weight_1)
print("weight_2", weight_2)

print("==========================")


# 选择表示最高概率的列
best_preds = np.asarray([np.argmax(line) for line in preds])
print(best_preds)

# 模型评估
print(precision_score(y_test, best_preds, average='macro'))  # 精准率
print(recall_score(y_test, best_preds, average='macro'))  # 召回率

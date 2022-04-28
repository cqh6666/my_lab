# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     append_csv_data
   Description:   ...
   Author:        cqh
   date:          2022/4/26 11:03
-------------------------------------------------
   Change Activity:
                  2022/4/26:
-------------------------------------------------
"""
__author__ = 'cqh'

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import confusion_matrix

breast_cancer = load_breast_cancer()
data_x = breast_cancer.data
data_y = breast_cancer.target

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.25)

dtrain = xgb.DMatrix(x_train, label=y_train)
dtest = xgb.DMatrix(x_test, label=y_test)

params = {
    'booster': 'gbtree',
    'max_depth': 11,
    'min_child_weight': 7,
    'subsample': 1,
    'colsample_bytree': 0.7,
    'eta': 0.05,
    'objective': 'binary:logistic',
    'nthread': 1,
    'verbosity': 0,
    'eval_metric': 'logloss',
    'seed': 998,
    'tree_method': 'hist'
}
num_round = 100  # the number of training iterations

bst = xgb.train(params, dtrain, num_boost_round=num_round)

y_predict = bst.predict(dtest)

# roc
from sklearn.metrics import roc_curve

fpr, tpr, thersholds = roc_curve(y_test, y_predict)
# for i, value in enumerate(thersholds):
#     print("%f %f %f" % (fpr[i], tpr[i], value))

# auc
from sklearn.metrics import auc

roc_auc = auc(fpr, tpr)

# roc_auc_score
from sklearn.metrics import roc_auc_score

xgb_roc_auc_score = roc_auc_score(y_test, y_predict)

# 绘制roc曲线
import matplotlib.pyplot as plt

plt.plot(fpr, tpr, label='ROC (area = {0:.2f})'.format(roc_auc), lw=2)

plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
plt.ylim([-0.05, 1.05])

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')  # 可以使用中文，但需要导入一些库即字体
plt.title('ROC Curve')
plt.legend(loc="lower right")
# plt.savefig('./roc_auc_score.png')
# plt.show()


xgb_model = xgb.XGBClassifier(n_jobs=1).fit(x_train, y_train)
sk_xgb_predict = xgb_model.predict_proba(x_test)
sk_xgb_roc_auc_score = roc_auc_score(y_test, sk_xgb_predict[:, 1])

from sklearn.metrics import classification_report

sk_label_xgb_predict = xgb_model.predict(x_test)
report = classification_report(y_test, sk_label_xgb_predict)

# 可视化输出

# 叶子节点 每个样本在所有树中的叶子节点
y_predict_leaf = bst.predict(dtest, pred_leaf=True)

xgb.to_graphviz(bst, num_trees=0)

# -*- coding: utf-8 -*-

# bayes_opt
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score
import lightgbm as lgb

# lgb_cv 函数定义了要去调哪些参数，并且使用交叉验证去计算特定指标的值（例子中用的是roc_auc)。
# 实际调参的时候也不可能所有参数一起调整，最好是分批次调参。
# 而且调参的时候可以观察本轮最优参数是否逼近了设定的搜索阈值，下一次调参的时候可以把搜索的范围扩大。

from utils.data_utils import get_train_test_X_y


def lgb_cv(n_estimators, min_gain_to_split, subsample, max_depth, colsample_bytree, min_child_samples, reg_alpha,
           reg_lambda, num_leaves, learning_rate):
    model = lgb.LGBMClassifier(boosting_type='gbdt', objective='binary', n_jobs=-1,
                               colsample_bytree=float(colsample_bytree),
                               min_child_samples=int(min_child_samples),
                               n_estimators=int(n_estimators),
                               num_leaves=int(num_leaves),
                               reg_alpha=float(reg_alpha),
                               reg_lambda=float(reg_lambda),
                               max_depth=int(max_depth),
                               subsample=float(subsample),
                               min_gain_to_split=float(min_gain_to_split),
                               learning_rate=float(learning_rate),
                               )
    cv_score = cross_val_score(model, train_x, train_y, scoring="roc_auc", cv=5).mean()
    return cv_score


is_engineer = True
is_norm = True
train_x, test_x, train_y, test_y = get_train_test_X_y(is_engineer, is_norm)
print('train x shape:', train_x.shape, 'test x shape:', test_x.shape)
print("is_engineer", is_engineer, "is_norm", is_norm)


# 实例化BayesianOptimization类，参数靠自己去定义取值范围
lgb_bo = BayesianOptimization(
    lgb_cv,
    {
        'colsample_bytree': (0.5, 1),
        'min_child_samples': (2, 200),
        'num_leaves': (5, 1000),
        'subsample': (0.5, 1),
        'max_depth': (2, 15),
        'n_estimators': (10, 1000),
        'reg_alpha': (0, 10),
        'reg_lambda': (0, 10),
        'min_gain_to_split': (0, 1),
        'learning_rate': (0, 1)
    },
)

# 训练
lgb_bo.maximize()

# 可以输出最优的值以及最优参数等等
print(lgb_bo.max)
res = lgb_bo.max
params_max = res['params']
params_max['n_estimators'] = int(params_max['n_estimators'])
params_max['min_child_samples'] = int(params_max['min_child_samples'])
params_max['num_leaves'] = int(params_max['num_leaves'])
params_max['max_depth'] = int(params_max['max_depth'])
params_max['reg_alpha'] = float(params_max['reg_alpha'])
params_max['colsample_bytree'] = float(params_max['colsample_bytree'])
params_max['reg_lambda'] = float(params_max['reg_lambda'])
params_max['subsample'] = float(params_max['subsample'])
params_max['min_gain_to_split'] = float(params_max['min_gain_to_split'])
params_max['learning_rate'] = float(params_max['learning_rate'])
print(params_max)

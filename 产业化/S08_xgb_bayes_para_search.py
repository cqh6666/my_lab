# -*- coding: utf-8 -*-
import xgboost as xgb
from bayes_opt import BayesianOptimization

# booster num

from utils.data_utils import get_train_test_X_y

max_booster_num = 1000
print("Max booster num:", max_booster_num)

# bayes search
bayes_search = True
init_points, n_iter = 500, 200
bayes_search_base_param = {
    "objective": 'binary:logistic',
    'eval_metric': 'auc',
    'tree_method': 'hist',
    "seed": 1001,
    "learning_rate": 0.05,
    # 'max_depth': 10,
    # 'min_child_weight': 6,
    # 'gamma': 1.8,
    # "subsample": 0.95,
    # "colsample_bytree": 0.52,
    # 'reg_lambda': 0.43,
    # 'reg_alpha': 1.36,
    # 'scale_pos_weight': 1,
    # 'max_delta_step': 1.64,

}
# ------------------------------------------------


def xgb_cv(
      max_depth=10,
      min_child_weight=5,
      gamma=0,
      subsample=1.0,
      colsample_bytree=1.0,
      reg_lambda=1,
      reg_alpha=0,
      max_delta_step=0,
      scale_pos_weight=1,
      learning_rate=0.1,
):
    global bayes_search_base_param
    bayes_search_base_param['max_depth'] = int(max_depth)
    bayes_search_base_param['min_child_weight'] = int(min_child_weight)
    bayes_search_base_param['gamma'] = float(gamma)
    bayes_search_base_param['subsample'] = float(subsample)
    bayes_search_base_param['colsample_bytree'] = float(colsample_bytree)
    bayes_search_base_param['reg_lambda'] = float(reg_lambda)
    bayes_search_base_param['reg_alpha'] = float(reg_alpha)
    bayes_search_base_param['max_delta_step'] = float(max_delta_step)
    bayes_search_base_param['scale_pos_weight'] = float(scale_pos_weight)
    bayes_search_base_param['learning_rate'] = float(learning_rate)

    cv = xgb.cv(params=bayes_search_base_param,
                dtrain=dtrain,
                num_boost_round=max_booster_num,
                early_stopping_rounds=5,
                nfold=5,
                metrics='auc',
                maximize=True,
                shuffle=True,
                stratified=False,
                verbose_eval=False
                )
    return cv["test-auc-mean"].values[-1]


def xgb_predict(
                max_depth=10,
                min_child_weight=5,
                gamma=0,
                subsample=1.0,
                colsample_bytree=1.0,
                reg_lambda=1,
                reg_alpha=0,
                max_delta_step=0,
                scale_pos_weight=1,
                learning_rate=0.1,
               ):

    global bayes_search_base_param
    bayes_search_base_param['max_depth'] = int(max_depth)
    bayes_search_base_param['min_child_weight'] = int(min_child_weight)
    bayes_search_base_param['gamma'] = float(gamma)
    bayes_search_base_param['subsample'] = float(subsample)
    bayes_search_base_param['colsample_bytree'] = float(colsample_bytree)
    bayes_search_base_param['reg_lambda'] = float(reg_lambda)
    bayes_search_base_param['reg_alpha'] = float(reg_alpha)
    bayes_search_base_param['max_delta_step'] = float(max_delta_step)
    bayes_search_base_param['scale_pos_weight'] = float(scale_pos_weight)
    bayes_search_base_param['learning_rate'] = float(learning_rate)

    eval_result = {}
    model = xgb.train(params=bayes_search_base_param,
                      dtrain=dtrain,
                      evals=[(dtest, 'test')],
                      early_stopping_rounds=5,
                      num_boost_round=max_booster_num,
                      evals_result=eval_result,
                      verbose_eval=False)
    print("cur best iteration is", (best_iter := model.best_iteration) + 1,
          "cur best auc is", auc := eval_result['test']['auc'][best_iter])
    return auc


def bayes_search_param():
    func = xgb_predict
    bounds = {
             'max_depth': (3, 10), # 8
             'min_child_weight': (1, 10), # 10
             'gamma': (0, 5), # 5
             'subsample': (0.5, 1.0), # 5
             'colsample_bytree': (0.5, 1.0), # 5
             'reg_lambda': (0, 5),
             'reg_alpha': (0, 5),
             'max_delta_step': (0, 10),
             'scale_pos_weight': (1, 7),
             'learning_rate': (0.01, 0.2),
    }

    opt_res = BayesianOptimization(f=func, pbounds=bounds, random_state=1001, verbose=0)
    print("bayes search bounds:\n", bounds)
    print("init_points=", init_points, "n_iter=", n_iter)
    opt_res.maximize(init_points=init_points, n_iter=n_iter)
    result = opt_res.max
    return result['params'], result['target']


# -------------------------------------------------------------------------------------------------------
# train_x = pd.DataFrame()
# train_y = pd.Series()
# test_x = pd.DataFrame()
# test_y = pd.Series()


is_engineer = False
is_norm = True
train_data_x, test_data_x, train_data_y, test_data_y = get_train_test_X_y(is_engineer, is_norm)
print('train x shape:', train_data_x.shape, 'test x shape:', test_data_x.shape)
print("is_engineer", is_engineer, "is_norm", is_norm)

dtrain = xgb.DMatrix(train_data_x, label=train_data_y)
dtest = xgb.DMatrix(test_data_x, label=test_data_y)

# ------- bayes search -------
print("-------Bayes search para-------")
print("bayes search base param:\n", bayes_search_base_param)
best_param, best_auc = bayes_search_param()

print(best_param, best_auc)
# ----------------------------

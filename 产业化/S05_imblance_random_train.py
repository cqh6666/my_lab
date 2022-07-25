# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     model_train
   Description:   ...
   Author:        cqh
   date:          2022/7/21 16:42
-------------------------------------------------
   Change Activity:
                  2022/7/21:
-------------------------------------------------
"""
__author__ = 'cqh'

import json

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import lightgbm as lgb
import warnings
from sklearn.preprocessing import MinMaxScaler
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek

from get_result_info import *

warnings.filterwarnings('ignore')


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def get_score(key, model, train_x, train_y, test_x, test_y):
    model.fit(train_x, train_y)
    y_score = model.predict_proba(test_x)[:, 1]
    y_true = test_y.to_numpy()
    auc_info = get_auroc(y_true, y_score)
    prc_info = get_auprc(y_true, y_score)
    gini_info = get_gini(y_true, y_score)
    psi_info = get_psi(y_true, y_score)
    # csi_info = get_csi(y_true, y_score)
    ks_info = get_ks(y_true, y_score)
    result_dict = {
        "auc_info": auc_info,
        "prc_info": prc_info,
        "gini_info": gini_info,
        "psi_info": psi_info,
        "ks_info": ks_info
    }
    result_json = json.dumps(result_dict, cls=NumpyEncoder)
    with open(f'./all_result_info/{key}/model_{key}_result_info.json', 'w') as f:
        f.write(result_json)

    return result_json


def save_csi_result(train_x, test_x):
    csi_dict = get_csi(train_x, test_x)
    with open(f'./all_result_info/csi_result.json', 'w') as f:
        f.write(json.dumps(csi_dict))
    print("save success!")


def plot_auroc(model_flag):
    with open(f"./all_result_info/{model_flag}/model_{model_flag}_result_info.json", 'r') as load_f:
        load_dict = json.load(load_f)

    fpr, tpr = load_dict['auc_info']['plot']
    roc_auc = load_dict['auc_info']['value']
    plt.plot(fpr, tpr, label='ROC (area={0:.2f})'.format(roc_auc), drawstyle="steps-post")
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.title("AUROC")
    plt.savefig(f'./all_result_info/{model_flag}/{model_flag}_auroc_result.png')
    plt.close()


def plot_auprc(model_flag):
    with open(f"./all_result_info/{model_flag}/model_{model_flag}_result_info.json", 'r') as load_f:
        load_dict = json.load(load_f)

    recall, precision = load_dict['prc_info']['plot']
    plt.plot(recall, precision, drawstyle="steps-post")
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title("AUPRC")
    plt.savefig(f'./all_result_info/{model_flag}/{model_flag}_auprc_result.png')
    plt.close()


def plot_gini(model_flag):
    with open(f"./all_result_info/{model_flag}/model_{model_flag}_result_info.json", 'r') as load_f:
        load_dict = json.load(load_f)

    x_values, (y_values, diagonal) = load_dict['gini_info']['plot']
    plt.stackplot(x_values, y_values, diagonal)
    plt.xlabel('x_values')
    plt.title("GINI")
    plt.savefig(f'./all_result_info/{model_flag}/{model_flag}_gini_result.png')
    plt.close()


def plot_ks(model_flag):
    with open(f"./all_result_info/{model_flag}/model_{model_flag}_result_info.json", 'r') as load_f:
        load_dict = json.load(load_f)

    thresholds, (fpr, tpr, ks) = load_dict['ks_info']['plot']
    data_df = pd.DataFrame(index=thresholds, data={"fpr":fpr,"tpr":tpr,"ks":ks})
    data_df.plot()
    plt.xlabel('thresholds')
    plt.title("KS")
    plt.savefig(f'./all_result_info/{model_flag}/{model_flag}_ks_result.png')
    plt.close()


if __name__ == '__main__':
    # model_list = ['dt', 'lgb', 'xgb', 'lr', 'rf']
    # for model in model_list:
    #     plot_auroc(model)
    #     plot_auprc(model)
    #     plot_ks(model)
    #     plot_gini(model)

    all_data = pd.read_csv("./default of credit card clients.csv")

    all_data_x = all_data.drop(['default payment next month'], axis=1)
    all_data_y = all_data['default payment next month']
    #
    min_max = MinMaxScaler()
    numer_list = ['LIMIT_BAL', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
                  'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    all_data_x[numer_list] = pd.DataFrame(min_max.fit_transform(all_data_x[numer_list]), columns=numer_list)

    pipeline = Pipeline([('over', SMOTETomek(random_state=2022)),
                         ('under', RandomUnderSampler(random_state=2022))])
    all_data_x, all_data_y = pipeline.fit_resample(all_data_x, all_data_y)

    train_x, test_x, train_y, test_y = train_test_split(all_data_x, all_data_y, test_size=0.3, random_state=2022)

    model_dict = {
        # "dt": DecisionTreeClassifier(),
        # "lr": LogisticRegression(),
        "rf": RandomForestClassifier(),
        "xgb": xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss"),
        "lgb": lgb.LGBMClassifier(),
        # "mlp": MLPClassifier(),
        # "svc": SVC()
    }
    for key, value in model_dict.items():
        get_score(key, value, train_x, train_y, test_x, test_y)

# for key, model in model_dict.items():
#     model.fit(train_x, train_y)
#     y_score = model.predict_proba(test_x)[:, 1]
#     y_true = test_y.to_numpy()
#     fpr, tpr, thresholds = roc_curve(y_true, y_score)
#     AUC = auc(fpr, tpr)
#     print(key, AUC)
#     precision, recall, thresholds = precision_recall_curve(y_true, y_score)
#     AUPR = auc(recall, precision)
#     print(key, AUPR)

"""

json_load = json.loads(json_dump)
a_restored = np.asarray(json_load["a"])
print(a_restored)
print(a_restored.shape)

"""

# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     S10_calibration
   Description:   ...
   Author:        cqh
   date:          2022/7/26 15:09
-------------------------------------------------
   Change Activity:
                  2022/7/26:
-------------------------------------------------
"""
__author__ = 'cqh'

import os

from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt
from utils.score_utils import *

from utils.data_utils import get_model_dict, get_train_test_X_y




def plot_calibrate(describe, y_pred):
    plt.figure(figsize=(8, 8))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_pred, n_bins=10, strategy='quantile')

    ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
             label="%s" % (describe,))
    ax2.hist(y_pred, range=(0, 1), bins=10, label=describe,
             histtype="step", lw=2)

    # all_res_df.to_csv("./S10_calibration_v3.csv")
    title = "S10_{calibration_v00"
    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title(title)

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

    plt.savefig(f"./S10_result/{title}.png")
    plt.tight_layout()
    plt.show()


def plot_calibration_curve(est, name, fig_index):
    """Plot calibration curve for est w/o and with calibration. """
    # Calibrated with isotonic calibration
    isotonic = CalibratedClassifierCV(est, cv=2, method='isotonic')

    # Calibrated with sigmoid calibration
    sigmoid = CalibratedClassifierCV(est, cv=2, method='sigmoid')

    # Logistic regression with no calibration as baseline
    lr = LogisticRegression(C=1.)

    fig = plt.figure(fig_index, figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for clf, name in [(lr, 'Logistic'),
                      (est, name),
                      (isotonic, name + ' + Isotonic'),
                      (sigmoid, name + ' + Sigmoid')]:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        if hasattr(clf, "predict_proba"):
            prob_pos = clf.predict_proba(X_test)[:, 1]
        else:  # use decision function
            prob_pos = clf.decision_function(X_test)
            prob_pos = \
                (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())

        clf_score = brier_score_loss(y_test, prob_pos, pos_label=y.max())
        fraction_of_positives, mean_predicted_value = \
            calibration_curve(y_test, prob_pos, n_bins=10)

        ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                 label="%s (%1.3f)" % (name, clf_score))

        ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
                 histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

    plt.tight_layout()


if __name__ == '__main__':

    is_smote = 1
    # 原始数据
    X_train, X_test, y_train, y_test = get_train_test_X_y()
    # SMOTE处理
    if is_smote == 1:
        X_train, y_train = smote_process(X_train, y_train)

    random_state = 2022

    models = ['dt', 'lr', 'xgb', 'lgb', 'rf']
    model_list = get_model_dict(models)

    res_columns = ['auc', 'prc', 'gini', 'ks', 'brier_init_score', 'psi_fixed', 'psi_qua', 'brier_new_score']

    """
    version=1 brier_init_score
    version=2 增加brier_new_score
    version=3 用原始数据(没用合成列)
    """
    version = 1
    save_path = f"./all_scores/v{version}/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if is_smote == 1:
        """
        1: 不校准
        2: 后校准
        3: 先校准
        """
        strategy_select = [1, 2, 3]
        index_desc = ['fit', 'fit+calibration', 'calibration+fit']

        for select, desc in zip(strategy_select, index_desc):
            get_all_model_score(select, desc)
            print(select, desc, "done!")

        print("done!")
    else:
        get_all_model_score(1, "init")
        print("done!")
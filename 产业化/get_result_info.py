# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     get_result_info
   Description:   ...
   Author:        cqh
   date:          2022/7/22 18:04
-------------------------------------------------
   Change Activity:
                  2022/7/22:
-------------------------------------------------
"""
__author__ = 'cqh'

from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def get_auroc(y_true, y_scores):
    """
    得到AUC结果
    :param y_true: 真实值
    :param y_scores: 预测值（概率）
    :return:
    {
        "roc_plot":(fpr,tpr),
        "auroc":value
    }
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    AUC = auc(fpr, tpr)
    return {"plot": (fpr, tpr), "value": AUC}


def get_auprc(y_true, y_scores):
    """
    https://blog.csdn.net/zfhsfdhdfajhsr/article/details/115055779
    :param y_true:
    :param y_scores:
    :return:
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    AUPR = auc(recall, precision)
    # if plot
    # result = pd.DataFrame(data={"precision": precision}, index=recall)
    # result.plot(drawstyle="steps-post")
    return {"plot": (recall, precision), "value": AUPR}


def get_gini(actual, pred):
    def gini(y_true, y_scores):
        """
        https://www.kaggle.com/code/batzner/gini-coefficient-an-intuitive-explanation
        :param y_true:
        :param y_scores:
        :return:
        """
        assert (len(y_true) == len(y_scores))
        all = np.asarray(np.c_[y_true, y_scores, np.arange(len(y_true))], dtype=np.float32)
        all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
        totalLosses = all[:, 0].sum()
        giniSum = all[:, 0].cumsum().sum() / totalLosses

        giniSum -= (len(y_true) + 1) / 2.
        return giniSum / len(y_true)

    gini_norm = gini(actual, pred) / gini(actual, actual)

    return {"plot": (get_gini_point(actual, pred)), "value": gini_norm}


def get_ks(y_true, y_scores):
    """
    KS（Kolmogorov-Smirnov）
    KS统计量是信用评分和其他很多学科中常见的统计量
    在金融风控领域中，常用于衡量模型对正负样本的区分度。通常来说，值越大，模型区分正负样本的能力越强
    一般0.3以上，说明模型的效果比较好（申请评分卡）。其定义如下：
    :param y_true:
    :param y_scores:
    :return:
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    ks = tpr - fpr
    KS_value = max(tpr - fpr)
    # if plot
    # result = pd.DataFrame(index=thresholds, data={"fpr": fpr, "tpr": tpr, "ks":ks})
    # result.plot()
    thresholds[0] = 1
    return {"plot": (thresholds, (fpr, tpr, ks)), "value": KS_value}


def get_psi(y_true, y_scores):
    """
    https://zhuanlan.zhihu.com/p/79682292
    https://towardsdatascience.com/checking-model-stability-and-population-shift-with-psi-and-csi-6d12af008783
    :return:
    """
    return {"value": np.mean(psi(y_true, y_scores))}


def get_csi(train_x, test_x):
    columns = train_x.columns
    result_dict = {}
    for col in columns:
        csi_values = get_psi(train_x[col].values, test_x[col].values)
        result_dict[col] = csi_values
    return {"value": result_dict}


def get_gini_point(y_true, y_scores):
    """https://www.kaggle.com/code/batzner/gini-coefficient-an-intuitive-explanation/notebook"""
    sorted_data = sorted(zip(y_true, y_scores), key=lambda d: d[1])
    sorted_actual = [d[0] for d in sorted_data]
    cumulative_actual = np.cumsum(sorted_actual)
    cumulative_index = np.arange(1, len(cumulative_actual) + 1)

    cumulative_actual_shares = cumulative_actual / sum(y_true)
    cumulative_index_shares = cumulative_index / len(y_scores)

    x_values = [0] + list(cumulative_index_shares)
    y_values = [0] + list(cumulative_actual_shares)

    # Display the 45° line stacked on top of the y values
    diagonal = [x - y for (x, y) in zip(x_values, y_values)]

    return x_values, (y_values, diagonal)


def psi(score_initial, score_new, num_bins=10, mode='fixed'):
    eps = 1e-4

    # Sort the data
    score_initial.sort()
    score_new.sort()

    # Prepare the bins
    min_val = min(min(score_initial), min(score_new))
    max_val = max(max(score_initial), max(score_new))
    if mode == 'fixed':
        bins = [min_val + (max_val - min_val) * (i) / num_bins for i in range(num_bins + 1)]
    elif mode == 'quantile':
        bins = pd.qcut(score_initial, q=num_bins, retbins=True)[
            1]  # Create the quantiles based on the initial population
    else:
        raise ValueError(f"Mode \'{mode}\' not recognized. Your options are \'fixed\' and \'quantile\'")
    bins[0] = min_val - eps  # Correct the lower boundary
    bins[-1] = max_val + eps  # Correct the higher boundary

    # Bucketize the initial population and count the sample inside each bucket
    bins_initial = pd.cut(score_initial, bins=bins, labels=range(1, num_bins + 1))
    df_initial = pd.DataFrame({'initial': score_initial, 'bin': bins_initial})
    grp_initial = df_initial.groupby('bin').count()
    grp_initial['percent_initial'] = grp_initial['initial'] / sum(grp_initial['initial'])

    # Bucketize the new population and count the sample inside each bucket
    bins_new = pd.cut(score_new, bins=bins, labels=range(1, num_bins + 1))
    df_new = pd.DataFrame({'new': score_new, 'bin': bins_new})
    grp_new = df_new.groupby('bin').count()
    grp_new['percent_new'] = grp_new['new'] / sum(grp_new['new'])

    # Compare the bins to calculate PSI
    psi_df = grp_initial.join(grp_new, on="bin", how="inner")

    # Add a small value for when the percent is zero
    psi_df['percent_initial'] = psi_df['percent_initial'].apply(lambda x: eps if x == 0 else x)
    psi_df['percent_new'] = psi_df['percent_new'].apply(lambda x: eps if x == 0 else x)

    # Calculate the psi
    psi_df['psi'] = (psi_df['percent_initial'] - psi_df['percent_new']) * np.log(
        psi_df['percent_initial'] / psi_df['percent_new'])

    # Return the psi values
    return psi_df['psi'].values


if __name__ == '__main__':
    X, y = load_iris(return_X_y=True)

    # Add noisy features
    random_state = np.random.RandomState(0)
    n_samples, n_features = X.shape
    X = np.concatenate([X, random_state.randn(n_samples, 200 * n_features)], axis=1)

    # Limit to the two first classes, and split into training and test
    X_train, X_test, y_train, y_test = train_test_split(
        X[y < 2], y[y < 2], test_size=0.5, random_state=random_state
    )

    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    y_predict = lr.predict_proba(X_test)[:, 1]

    result1 = get_auroc(y_test, y_predict)
    result2 = get_auprc(y_test, y_predict)
    result3 = get_ks(y_test, y_predict)
    result4 = get_gini(y_test, y_predict)
    result6 = get_psi(y_test, y_predict)
    x_train_df = pd.DataFrame(X_train)
    x_test_df = pd.DataFrame(X_test)
    result7 = get_csi(x_train_df, x_test_df)

    # fpr, tpr = result2['plot']
    # plt.plot(fpr, tpr, drawstyle="steps-post")
    # plt.xlabel('recall')
    # plt.ylabel('precision')

    # x_values, (y_values, diagonal) = result4['plot']
    # plt.stackplot(x_values, y_values, diagonal)
    # plt.show()

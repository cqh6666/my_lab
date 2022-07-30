# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     plto_utils
   Description:   ...
   Author:        cqh
   date:          2022/7/28 22:22
-------------------------------------------------
   Change Activity:
                  2022/7/28:
-------------------------------------------------
"""
__author__ = 'cqh'

import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms

def plot_auroc(save_file, x, y):
    """
    :param result_list: 绘制列表 [ {}, {}, {}, {} ]
    :param save_file:
    :return:
    """
    fpr, tpr = x, y
    plt.plot(fpr, tpr, drawstyle="steps-post")
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.title("AUROC")
    plt.savefig(save_file)
    plt.close()


def plot_auprc(save_file, x, y):
    recall, precision = x, y
    plt.plot(recall, precision, drawstyle="steps-post")
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title("AUPRC")
    plt.savefig(save_file)
    plt.close()


def plot_gini(save_file, x, y):
    x_values, (y_values, diagonal) = x, y
    plt.stackplot(x_values, y_values, diagonal)
    plt.xlabel('x_values')
    plt.title("GINI")
    plt.savefig(save_file)
    plt.close()


def plot_ks(save_file, x, y):
    thresholds, (fpr, tpr, ks) = x, y
    data_df = pd.DataFrame(index=thresholds, data={"fpr": fpr, "tpr": tpr, "ks": ks})
    data_df.plot()
    plt.xlabel('thresholds')
    plt.title("KS")
    plt.savefig(save_file)
    plt.close()


def plot_calibration_curve(save_file, x, y):
    mean_predicted_value, fraction_of_positives = x, y
    fig, ax = plt.subplots()
    # only these two lines are calibration curves
    ax.plot(mean_predicted_value, fraction_of_positives, 's-', label='best_xgb+calibration+fit')
    # reference line, legends, and axis labels
    ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    fig.suptitle('Calibration plots  (reliability curve)')
    ax.set_xlabel('Mean predicted value')
    ax.set_ylabel('Fraction of positives')
    plt.savefig(save_file)
    plt.close()


def plot_psi(save_file, x, y):
    labels, (first, second) = x, y
    x = np.arange(len(labels))  # x轴刻度标签位置
    width = 0.25  # 柱子的宽度
    # 计算每个柱子在x轴上的位置，保证x轴刻度标签居中
    # x - width/2，x + width/2即每组数据在x轴上的位置
    plt.bar(x - width / 2, first, width, label='init')
    plt.bar(x + width / 2, second, width, label='pred')
    plt.ylabel('percent')
    plt.title('Population Stability Index By Every Bin')
    # x轴刻度标签位置不进行计算
    plt.xticks(x, labels=labels)
    plt.savefig(save_file)
    plt.close()

# -*- coding: gbk -*-
"""
-------------------------------------------------
   File Name:     0010_get_auc_result
   Description:   ...
   Author:        cqh
   date:          2022/4/20 20:40
-------------------------------------------------
   Change Activity:
                  2022/4/20:
-------------------------------------------------
"""
__author__ = 'cqh'

import pandas as pd
import os
from sklearn.metrics import roc_auc_score
import numpy as np
import sys
import time
from my_logger import MyLog


def process_and_plot_result():
    """
    ����auc��� �����Ƴ�ͼ��
    :return:
    """
    all_auc_result = pd.read_csv(auc_result_file)
    all_auc_result['global'] = 0.814673068632535
    all_auc_result['sub_global'] = 0.7721542229664811
    all_auc_result.sort_values(by=['iter_idx'], inplace=True)
    ax = all_auc_result.plot(x='iter_idx', y=['auc', 'global', 'sub_global'], title=f"auc_result_{transfer_flag}")
    fig = ax.get_figure()
    png_file_name = os.path.join(AUC_RESULT_PATH, f"24h_lr_old_auc_result_plot_{transfer_flag}.png")
    fig.savefig(png_file_name)
    print("save auc result to png success!")


if __name__ == '__main__':
    # input params
    pre_hour = 24
    is_transfer = 1
    # �Ƿ�Ǩ�ƣ���Ӧ��ͬ·��
    transfer_flag = "transfer" if is_transfer == 1 else "no_transfer"
    AUC_RESULT_PATH = './'
    # �����յ�auc������б���
    auc_result_file = os.path.join(AUC_RESULT_PATH, f"{pre_hour}h_lr_old_auc_result_{transfer_flag}.csv")
    process_and_plot_result()
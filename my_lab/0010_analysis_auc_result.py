# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     analysis_auc_result
   Description:   ...
   Author:        cqh
   date:          2022/5/9 9:42
-------------------------------------------------
   Change Activity:
                  2022/5/9:
-------------------------------------------------
"""
__author__ = 'cqh'

import pandas as pd
import matplotlib.pyplot as plt

def get_auc():
    with open(PATH, "r") as f:
        iter_idx = []
        auc = []
        for line in f.readlines():
            line_str = line.split(',')
            iter_idx.append(line_str[0])
            auc.append(line_str[1])

        data = {'iter': iter_idx, 'auc': auc}
        df = pd.DataFrame(data)
        df['iter'] = df['iter'].astype(int)
        df['auc'] = df['auc'].astype(float)
        print(df)
        return df


def plot_auc(all_auc_result):
    all_auc_result.sort_values(by=['iter'], inplace=True)
    all_auc_result.plot(x='iter', y='auc')
    plt.show()


if __name__ == '__main__':
    PATH = "../other_file/auc_result.txt"

    result = get_auc()
    plot_auc(result)

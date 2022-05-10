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

PATH = "auc_result.txt"


def get():
    with open(PATH, "r") as f:
        iter = []
        auc = []
        for line in f.readlines():
            iter.append(line[line.index('[') + 1:line.index(']')])
            auc.append(line[line.index('-') + 1:].strip())

        data = {'iter': iter, 'auc': auc}
        df = pd.DataFrame(data)
        df['iter'] = df['iter'].astype(int)
        df['auc'] = df['auc'].astype(float)
        print(df)
        return df


if __name__ == '__main__':
    get()

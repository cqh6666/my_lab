# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     plot_result
   Description:   ...
   Author:        cqh
   date:          2022/7/7 15:12
-------------------------------------------------
   Change Activity:
                  2022/7/7:
-------------------------------------------------
"""
__author__ = 'cqh'

import pandas as pd

# result = pd.read_csv(r"./S05_ev_r.csv")


ax = result.plot(y=['ev_r', 'ev_r_sum'], title='ev_ratio')
fig = ax.get_figure()
png_file_name = f"./ev_r_sum_result.png"
fig.savefig(png_file_name)
print("save auc result to png success!")

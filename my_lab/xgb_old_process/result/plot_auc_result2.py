# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     plot_auc_result
   Description:   ...
   Author:        cqh
   date:          2022/6/20 9:25
-------------------------------------------------
   Change Activity:
                  2022/6/20:
-------------------------------------------------
"""
__author__ = 'cqh'

import os

import pandas as pd

index = [0, 5, 10, 15, 20]

select5_auc_transfer = [0.874638, 0.873861, 0.873902, 0.874417, 0.875079]
select10_auc_transfer = [0.875822, 0.874767, 0.8748, 0.8749, 0.874406]
select20_auc_transfer = [0.876405, 0.87664, 0.876942, 0.876124, 0.875901]

select5_auc_no_transfer = [0.848808, 0.848513, 0.847813, 0.847527, 0.846504]
select10_auc_no_transfer = [0.853628, 0.852764, 0.851812, 0.851276, 0.851607]
select20_auc_no_transfer = [0.851796, 0.853109, 0.853151, 0.852738, 0.853344]

data_dict = {
    "iter_idx": index,
    "select5_auc_no_transfer": select5_auc_no_transfer,
    "select10_auc_no_transfer": select10_auc_no_transfer,
    "select20_auc_no_transfer": select20_auc_no_transfer
}
auc_df = pd.DataFrame(data_dict)
x_label = 'iter_idx'
y_label = list(data_dict.keys())[1:]
title = "select_auc_no_transfer"
png_file_name = f"select_auc_no_transfer.png"

ax = auc_df.plot(x='iter_idx', y=y_label, title=title)
fig = ax.get_figure()
fig.savefig(png_file_name)
print("save auc result to png success!")

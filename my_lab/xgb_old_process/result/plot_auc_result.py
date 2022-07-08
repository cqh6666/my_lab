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

weight_auc_transfer = [0.875822, 0.874767, 0.8748, 0.8749, 0.874406]
shap_auc_transfer = [0.876559, 0.876622, 0.876368, 0.876809, 0.875303]

weight_auc_no_transfer = [0.853628, 0.852764, 0.851812, 0.851276, 0.851607]
shap_auc_no_transfer = [0.85357, 0.852141, 0.853937, 0.853587, 0.853571]

weight_local25_auc_transfer = [0.875669, 0.875393, 0.87567, 0.875629, 0.875055]
weight_local50_auc_transfer = [0.875822, 0.874767, 0.8748, 0.8749, 0.874406]
weight_local100_auc_transfer = [0.874661, 0.872767, 0.872938, 0.873894, 0.873999]

# auc_df = pd.DataFrame({"iter_idx":index, "weight_auc_transfer":weight_auc_transfer, "shap_auc_transfer": shap_auc_transfer})
# x_label = 'iter_idx'
# y_label = ['weight_auc_transfer', 'shap_auc_transfer']
# title = "weight_shap_auc_transfer"
# ax = auc_df.plot(x='iter_idx', y=y_label, title=title)
# fig = ax.get_figure()
# png_file_name = f"weight_shap_auc_transfer.png"
# fig.savefig(png_file_name)
# print("save auc result to png success!")

# auc_df = pd.DataFrame({"iter_idx":index, "weight_auc_no_transfer":weight_auc_no_transfer, "shap_auc_no_transfer": shap_auc_no_transfer})
# x_label = 'iter_idx'
# y_label = ['weight_auc_no_transfer', 'shap_auc_no_transfer']
# title = "weight_shap_auc_no_transfer"
# png_file_name = f"weight_shap_auc_no_transfer.png"
#
# ax = auc_df.plot(x='iter_idx', y=y_label, title=title)
# fig = ax.get_figure()
# fig.savefig(png_file_name)
# print("save auc result to png success!")

auc_df = pd.DataFrame({"iter_idx":index,
                       "weight_local25_auc_transfer":weight_local25_auc_transfer,
                       "weight_local50_auc_transfer": weight_local50_auc_transfer,
                       "weight_local100_auc_transfer": weight_local100_auc_transfer})
x_label = 'iter_idx'
y_label = ['weight_local25_auc_transfer', 'weight_local50_auc_transfer', 'weight_local100_auc_transfer']
title = "weight_local_auc_transfer"
png_file_name = f"weight_local_auc_transfer.png"

ax = auc_df.plot(x='iter_idx', y=y_label, title=title)
fig = ax.get_figure()
fig.savefig(png_file_name)
print("save auc result to png success!")

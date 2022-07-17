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
import matplotlib.pyplot as plt

model = "LR"
transfer_flag = "no_transfer"
csv_file_name = f"./csv/S03_{model}_PCA.csv"
# csv_file_name = f"./S01_{model}_{transfer_flag}.csv"
png_file_name = f"S03_{model}_PCA_AUC_diff_components"
png_save_path = "./png/" + png_file_name

result = pd.read_csv(csv_file_name, index_col=0)
ax = result.plot(y=result.columns, title=png_file_name)
fig = ax.get_figure()
fig.savefig(png_save_path)
print("save auc result to png success!")

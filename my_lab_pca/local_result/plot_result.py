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

# 设置中文字体和负号正常显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


file = "S07_test_dim100_v1"
csv_file_name = f"./csv/{file}.csv"
png_save_path = "./png/" + file

result = pd.read_csv(csv_file_name, index_col=0)
ax = result.plot(y=result.columns, title="LR迭代降维匹配性能对比")
plt.axhline(y=0.829691, color='black', linestyle="dashed", linewidth=2)
plt.axhline(y=0.765258, color='black', linestyle="dashed", linewidth=2)
plt.xlabel("迭代次数")
plt.ylabel("AUC")
plt.savefig(png_save_path)
print("save auc result to png success!")

# file = "S03_LR_PCA_DIFF"
# csv_file_name = f"./csv/{file}.csv"
# png_save_path = "./png/" + file
#
# result = pd.read_csv(csv_file_name, index_col=0)
# result.drop([result.columns.tolist()[-1]], axis=1)
# ax = result.plot(y=result.columns, title="LR不同维度降维匹配性能对比")
# plt.axhline(y=0.829691, color='black', linestyle="dashed", linewidth=2)
# plt.axhline(y=0.765258, color='black', linestyle="dashed", linewidth=2)
# plt.xlabel("降维维度")
# plt.ylabel("AUC")
# plt.savefig(png_save_path)
# print("save auc result to png success!")
#
# file = "S04_XGB_KMEANS"
# csv_file_name = f"./csv/{file}.csv"
# png_save_path = "./png/" + file
# result = pd.read_csv(csv_file_name, index_col=0)
# ax = result.plot(y=result.columns, title="XGB不同相似样本均值匹配")
# plt.axhline(y=0.875523, color='black', linestyle="dashed", linewidth=2)
# plt.axhline(y=0.8264, color='black', linestyle="dashed", linewidth=2)
#
# plt.xlabel("均值样本数")
# plt.ylabel("AUC")
# plt.savefig(png_save_path)
# print("save auc result to png success!")
#
# file = "S04_LR_KMEANS"
# csv_file_name = f"./csv/{file}.csv"
# png_save_path = "./png/" + file
# result = pd.read_csv(csv_file_name, index_col=0)
# ax = result.plot(y=result.columns, title="LR不同相似样本均值匹配")
# plt.axhline(y=0.829691, color='black', linestyle="dashed", linewidth=2)
# plt.axhline(y=0.765258, color='black', linestyle="dashed", linewidth=2)
# plt.xlabel("均值样本数")
# plt.ylabel("AUC")
# plt.savefig(png_save_path)
# print("save auc result to png success!")

# file = "S06_pca_psm_XGB"
# csv_file_name = f"./csv/{file}.csv"
# png_save_path = "./png/" + file
# result = pd.read_csv(csv_file_name, index_col=0)
# ax = result.plot(y=result.columns, title="XGB是否使用相似性度量进行降维匹配性能对比")
# plt.axhline(y=0.875523, color='black', linestyle="dashed", linewidth=2)
# plt.axhline(y=0.8264, color='black', linestyle="dashed", linewidth=2)
#
# plt.xlabel("降维维度")
# plt.ylabel("AUC")
# plt.savefig(png_save_path)
# print("save auc result to png success!")
#
# file = "S06_pca_psm_LR"
# csv_file_name = f"./csv/{file}.csv"
# png_save_path = "./png/" + file
#
# result = pd.read_csv(csv_file_name, index_col=0)
# ax = result.plot(y=result.columns, title="LR是否使用相似性度量进行降维匹配性能对比")
# plt.axhline(y=0.829691, color='black', linestyle="dashed", linewidth=2)
# plt.axhline(y=0.765258, color='black', linestyle="dashed", linewidth=2)
# plt.xlabel("将为维度")
# plt.ylabel("AUC")
# plt.savefig(png_save_path)
# print("save auc result to png success!")
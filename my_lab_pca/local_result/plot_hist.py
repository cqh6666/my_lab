# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     plot_hist
   Description:   ...
   Author:        cqh
   date:          2022/8/15 16:32
-------------------------------------------------
   Change Activity:
                  2022/8/15:
-------------------------------------------------
"""
__author__ = 'cqh'

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np

# 这两行代码解决 plt 中文显示的问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

csv_file_name = "./csv/S06_pca_psm.csv"
result = pd.read_csv(csv_file_name)

# 输入统计数据
waters = result.index.tolist()
buy_number_male = result['psm0_tra0'].values
buy_number_female = result['psm1_tra0'].values

bar_width = 0.3  # 条形宽度
index_male = np.arange(len(waters))  # 男生条形图的横坐标
index_female = index_male + bar_width  # 女生条形图的横坐标

# 使用两次 bar 函数画出两组条形图
plt.bar(index_male, height=buy_number_male, width=bar_width, color='b', label='男性')
plt.bar(index_female, height=buy_number_female, width=bar_width, color='g', label='女性')

plt.legend()  # 显示图例
plt.xticks(index_male + bar_width/2, waters)  # 让横坐标轴刻度显示 waters 里的饮用水， index_male + bar_width/2 为横坐标轴刻度的位置
plt.ylabel('购买量')  # 纵坐标轴标题
plt.title('购买饮用水情况的调查结果')  # 图形标题

plt.show()
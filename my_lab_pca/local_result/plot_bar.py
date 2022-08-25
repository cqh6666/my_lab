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
import numpy as np

# 设置中文字体和负号正常显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# plt.figure(figsize=(12,6)) #建立图形
# name = ['%d月'%x for x in range(1,13)]   #创建月份
# value1 = [np.random.randint(10000) for i in range(12)]   #创建随机数字
# value1.sort()    #对数字进行排序
# value2 = [np.random.randint(10000) for i in range(12)]   #创建随机数字
# value2.sort()    #对数字进行排序
# x = range(12)

"""
绘制条形图
left:长条形中点横坐标
height:长条形高度
width:长条形宽度，默认值0.8
alpha:透明度
color:颜色
label:标签，为后面设置legend准备
"""
# bar1 = plt.bar(left = [i - 0.2 for i in x], height = value1, width = 0.4,
#                alpha = 0.8, color = 'r',label = '一部门')                  #第一个图
#
# bar2 = plt.bar([i + 0.2 for i in x],value2,width = 0.4,
#                alpha = 0.8,color = 'g',label = '二部门')                   #第二个图
#
# plt.xticks(x,name)      #设置x轴刻度显示值
# plt.ylim(0,10500)       #y轴的范围
# plt.title('XXX公司')     #标题
# plt.xlabel('月份')       #x轴的标签
# plt.ylabel('数量')       #y轴的标签
# plt.legend()            #设置图例

'''
get_height:获取值
get_x：获取x轴的位置
get_width:获取图形的宽度
text(x,y,s,fontsize,ha,va)
    x,y:表示坐标值上的值
    s:表示说明文字
    fontsize:表示字体大小
    ha：垂直显示方式{'centee':'中心', 'right':'右', 'left':'左'}
    va：水平显示方式{'center':'中心', 'top':'下', 'bottom':'上', 'baseline':'基线'} 
'''

# for rect in bar1:
#     height = rect.get_height()  #获得bar1的高度
#     plt.text(rect.get_x() + rect.get_width() / 2, height+3, str(height), ha="center", va="bottom")
# for rect in bar2:
#     height = rect.get_height()
#     plt.text(rect.get_x() + rect.get_width() / 2, height+3, str(height), ha="center", va="bottom")
#
# plt.show()

# import pandas as pd
# file = "S06_xgb_no_transfer"
# data = pd.read_csv(f"./csv/{file}.csv")
# data.index = ["直接匹配","PCA降维(100)匹配","PCA降维(1000)匹配"]
# bar = data.plot(figsize=(6,5), kind='bar', ylim=[0.82, 0.87], rot=0)
# plt.axhline(y=0.8264, color='black', linestyle="dashed", linewidth=2)
#
# plt.title("XGB非迁移下不同初始相似性度量对比")
# plt.ylabel("AUC")
# plt.savefig(f"png/{file}")
# plt.show()

# import pandas as pd
# file = "S06_xgb_transfer"
# data = pd.read_csv(f"./csv/{file}.csv")
# data.index = ["直接匹配","PCA降维(100)匹配","PCA降维(1000)匹配"]
# bar = data.plot(figsize=(6,5), kind='bar', ylim=[0.84, 0.89], rot=0)
# plt.axhline(y=0.875523, color='black', linestyle="dashed", linewidth=2)
# plt.title("XGB迁移下不同初始相似性度量对比")
# plt.ylabel("AUC")
# plt.savefig(f"png/{file}")
# plt.show()

# import pandas as pd
# file = "S06_lr_no_transfer"
# data = pd.read_csv(f"./csv/{file}.csv")
# data.index = ["直接匹配","PCA降维(100)匹配","PCA降维(1000)匹配"]
# bar = data.plot(figsize=(6,5), kind='bar', ylim=[0.74, 0.80], rot=0)
# plt.axhline(y=0.765258, color='black', linestyle="dashed", linewidth=2)
# plt.title("LR非迁移下不同初始相似性度量对比")
# plt.ylabel("AUC")
# plt.savefig(f"png/{file}")
# plt.show()
#
# import pandas as pd
# file = "S06_lr_transfer"
# data = pd.read_csv(f"./csv/{file}.csv")
# data.index = ["直接匹配","PCA降维(100)匹配","PCA降维(1000)匹配"]
# bar = data.plot(figsize=(6,5), kind='bar', ylim=[0.81, 0.87], rot=0)
# plt.axhline(y=0.829691, color='black', linestyle="dashed", linewidth=2)
# plt.title("LR迁移下不同初始相似性度量对比")
# plt.ylabel("AUC")
# plt.savefig(f"png/{file}")
# plt.show()

# import pandas as pd
# file = "S08_AUTO_PCA_xgb_no_transfer"
# data = pd.read_csv(f"./csv/{file}.csv", index_col=0)
# data.index = ["降维(20)匹配","降维(100)匹配"]
# bar = data.plot(kind='bar', ylim=[0.82, 0.86], rot=0)
# plt.axhline(y=0.8264, color='black', linestyle="dashed", linewidth=2)
# plt.title("XGB非迁移下AutoEncoder与PCA降维性能对比")
# plt.ylabel("AUC")
# plt.savefig(f"png/{file}")
# plt.show()
#
# import pandas as pd
# file = "S08_AUTO_PCA_xgb_transfer"
# data = pd.read_csv(f"./csv/{file}.csv", index_col=0)
# data.index = ["降维(20)匹配","降维(100)匹配"]
# bar = data.plot(kind='bar', ylim=[0.84, 0.89], rot=0)
# plt.axhline(y=0.875523, color='black', linestyle="dashed", linewidth=2)
# plt.title("XGB迁移下AutoEncoder与PCA降维性能对比")
# plt.ylabel("AUC")
# plt.savefig(f"png/{file}")
# plt.show()




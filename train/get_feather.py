# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     get_feather
   Description:   ...
   Author:        cqh
   date:          2022/4/13 16:04
-------------------------------------------------
   Change Activity:
                  2022/4/13:
-------------------------------------------------
"""
__author__ = 'cqh'

import pandas as pd

from sklearn.datasets import load_iris


iris = load_iris()

df_x = pd.DataFrame(iris['data'], columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'])
df_y = pd.DataFrame(iris['target'], columns=['Species'])

dfy = df_y['Species']
all_classes = dfy.value_counts()

# df = pd.concat([df1, df2], axis=1)
#
df_x.to_feather('../feather/iris_data.feather')
# print("done!")
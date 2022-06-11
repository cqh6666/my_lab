# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     cal
   Description:   ...
   Author:        cqh
   date:          2022/6/9 19:32
-------------------------------------------------
   Change Activity:
                  2022/6/9:
-------------------------------------------------
"""
__author__ = 'cqh'
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
import pandas as pd

l500 = pd.read_csv(r"D:\lab\other_file\0006_24h_global_lr_400.csv").squeeze()
l400 = pd.read_csv(r"D:\lab\other_file\0006_24h_global_lr_500.csv").squeeze()
result = cosine_similarity([l500,l400])

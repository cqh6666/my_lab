# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     load_xgb_model
   Description:   ...
   Author:        cqh
   date:          2022/4/25 16:24
-------------------------------------------------
   Change Activity:
                  2022/4/25:
-------------------------------------------------
"""
__author__ = 'cqh'

import joblib
import pandas as pd

# load data
source_path = r"D:\lab\feather\iris_data.feather"
data = pd.read_feather(source_path)

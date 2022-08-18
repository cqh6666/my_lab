# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     output_csv
   Description:   ...
   Author:        cqh
   date:          2022/8/16 15:30
-------------------------------------------------
   Change Activity:
                  2022/8/16:
-------------------------------------------------
"""
__author__ = 'cqh'
import pandas as pd
import os

# autoEncoder 降维 v1 100维度
dimension = 100
version =2
encoder_path = "/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/code_pca/result/new_data/"
encoder_train_data_x = pd.read_csv(os.path.join(encoder_path, f"train_data_dim{dimension}_v{version}.csv"), index_col=0)
encoder_test_data_x = pd.read_csv(os.path.join(encoder_path, f"test_data_dim{dimension}_v{version}.csv"), index_col=0)
print(f"load encoder data {encoder_train_data_x.shape}, {encoder_test_data_x.shape}")
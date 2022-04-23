"""
evaluate personal xgb model with many csv files
"""
import pandas as pd
import os
from sklearn.metrics import roc_auc_score

# file_name prefix
file_name_prefix = '0009_proba_01_001_005-1-tran_'
# folder
folder = '/panfs/pfs.local/work/liu/xzhang_sta/tangxizhuo/pg/test_result/'

# init result
result = pd.DataFrame()

# traverse all files in current folder
for file_name in os.listdir(folder):
    # select related csv files
    if file_name_prefix in file_name:
        cur_path = f'{folder}{file_name}'
        # read csv with index
        cur_file = pd.read_csv(cur_path, index_col=0)
        result = pd.concat([result, cur_file], axis=0)

# sort row index
result.sort_index(axis=0, inplace=True)

# calculate roc and print
roc = roc_auc_score(result['real'], result['proba'])
print("roc:", roc)

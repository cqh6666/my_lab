# encoding=gbk
import multiprocessing as mp

import time
import numpy as np
import pandas as pd

def my_worker(dt):
    print(dt, mp.current_process().pid)
    time.sleep(10)


if __name__ == '__main__':
    file = '0006_xgb_global_feature_weight_importance_boost91_v0.csv'
    data = pd.read_csv(file).squeeze('columns')

    for idx,d in enumerate(data):
        print(idx,d)
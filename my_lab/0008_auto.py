"""
check metric KL learning
"""
import time
import os

iter_init = 5
os.system(f'sbatch /panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/code/0008_auto_learn_KL_use_XGB_mp.sh {iter_init}')

init_boost = 91
xgb_boost_num = 1
step = 5

while iter_init < 120:

    if os.path.exists(
        f'/panfs/pfs.local/work/liu/xzhang_sta/tangxizhuo/pg/KL_result/0008_24h_{iter_init}_feature_weight_initboost{init_boost}_localboost{xgb_boost_num}.csv'
    ):
        time.sleep(20)
        iter_init += step
        os.system(
            f'sbatch /panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/code/0008_auto_learn_KL_use_XGB_mp.sh {iter_init}')
    else:
        time.sleep(30)

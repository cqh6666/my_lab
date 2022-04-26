"""
check metric KL learning
"""
import time
import os

iter_init = 0
os.system(f'sbatch /panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/code/0008_auto_learn_KL_use_XGB_mt.sh {iter_init}')

init_boost = 91
xgb_boost_num = 50
step = 5

while iter_init < 121:

    if os.path.exists(f'/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/result/personal_model_with_xgb/0008_24h_{iter_init+step}_feature_weight_initboost{init_boost}_localboost{xgb_boost_num}.csv'):
        time.sleep(20)
        iter_init += step
        os.system(
            f'sbatch /panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/code/0008_auto_learn_KL_use_XGB_mp.sh {iter_init}')
    else:
        time.sleep(30)

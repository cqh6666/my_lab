"""
check metric KL learning
"""
import time
import os

iter_init = 0
os.system(f'sbatch /panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/code/0008_auto_learn_KL_use_XGB_mt_no_transfer.sh {iter_init}')

xgb_boost_num = 70
step = 2

while iter_init < 121:
    if os.path.exists(f'/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/result/personal_model_with_xgb/24h_xgb_model/24h_no_transfer_psm/0008_24h_{iter_init+step}_feature_weight_localboost{xgb_boost_num}_mt.csv'):
        time.sleep(20)
        iter_init += step
        os.system(
            f'sbatch /panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/code/0008_auto_learn_KL_use_XGB_mt_no_transfer.sh {iter_init}')
    else:
        time.sleep(30)

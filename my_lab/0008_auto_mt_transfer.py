"""
check metric KL learning
"""
import time
import os
import sys

iter_init = int(sys.argv[1])
os.system(f'sbatch /panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/code/0008_auto_learn_KL_use_XGB_mt_transfer.sh {iter_init}')

glo_tl_boost_num = 20
xgb_boost_num = 50
step = 2
pre_hour = 24

PSM_SAVE_PATH = f"/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/result/personal_model_with_xgb/{pre_hour}h_xgb_model/{pre_hour}h_transfer_psm/"

while iter_init < 121:
    wi_file_name = os.path.join(PSM_SAVE_PATH, f'0008_24h_{iter_init+step}_feature_weight_gtlboost{glo_tl_boost_num}_localboost{xgb_boost_num}.csv')
    if os.path.exists(wi_file_name):
        time.sleep(20)
        iter_init += step
        os.system(
            f'sbatch /panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/code/0008_auto_learn_KL_use_XGB_mt_transfer.sh {iter_init}')
    else:
        time.sleep(30)

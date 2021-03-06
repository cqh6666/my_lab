#enocding=gbk
"""
check metric KL learning
"""
import sys
import time
import os

iter_init = int(sys.argv[1])
os.system(f'sbatch /panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/code/0008_auto_learn_KL_use_XGB_mt_no_transfer.sh {iter_init}')

xgb_boost_num = 70
step = 2
pre_hour = 24

PSM_SAVE_PATH = f"/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/result/personal_model_with_xgb/{pre_hour}h_xgb_model/{pre_hour}h_no_transfer_psm/"

while iter_init < 121:
    wi_file_name = os.path.join(PSM_SAVE_PATH, f"0008_24h_{iter_init + step}_feature_weight_localboost{xgb_boost_num}_no_transfer.csv")
    if os.path.exists(wi_file_name):
        time.sleep(20)
        iter_init += step
        os.system(
            f'sbatch /panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/code/0008_auto_learn_KL_use_XGB_mt_no_transfer.sh {iter_init}')
    else:
        time.sleep(30)

"""
check metric KL learning
"""
import time
import os
import sys

is_transfer = int(sys.argv[1])
iter_init = int(sys.argv[2])
print(f"[params] is_transfer:{is_transfer}, iter_init:{iter_init}")

step = 10
pre_hour = 24

transfer_flag = "no_transfer" if is_transfer == 0 else "transfer"

root_dir = f"{pre_hour}h"
PSM_SAVE_PATH = f'/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/result/personal_model_with_lr/{root_dir}/psm_{transfer_flag}/'

if iter_init == 0:
    os.system(f'sbatch /panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/code/0008_learn_KL_use_LR.sh {is_transfer} {iter_init}')
    iter_init += step

while iter_init < 121:
    wi_file_name = os.path.join(PSM_SAVE_PATH, f"0008_{pre_hour}h_{iter_init}_psm_{transfer_flag}.csv")
    if os.path.exists(wi_file_name):
        time.sleep(10)
        os.system(
            f'sbatch /panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/code/0008_learn_KL_use_LR.sh {is_transfer} {iter_init}')
        iter_init += step
    else:
        time.sleep(30)

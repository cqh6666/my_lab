"""
check metric KL learning
"""
import time
import os
import sys

is_transfer = int(sys.argv[1])
iter_init = int(sys.argv[2])
print(f"[params] - is_transfer:{is_transfer}, iter_init:{iter_init}")

step = 15
n_components = 100
version = 1
transfer_flag = "no_transfer" if is_transfer == 0 else "transfer"

PSM_SAVE_PATH = f'./result/S06_temp/psm_{transfer_flag}'

if iter_init == 0:
    os.system(f'sbatch /panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/code_pca/code_lr/S06_learn_psm_with_pca.sh {is_transfer} {iter_init}')
    iter_init += step

while iter_init < 121:
    wi_file_name = os.path.join(PSM_SAVE_PATH, f"S06_iter{iter_init}_dim{n_components}_tra{is_transfer}_v{version}.csv")
    if os.path.exists(wi_file_name):
        time.sleep(5)
        os.system(
            f'sbatch /panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/code_pca/code_lr/S06_learn_psm_with_pca.sh {is_transfer} {iter_init}')
        print(f"iter:[{iter_init}] - exist psm file: {wi_file_name}")
        iter_init += step
    else:
        time.sleep(30)

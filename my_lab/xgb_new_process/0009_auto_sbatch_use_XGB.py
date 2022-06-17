#encoding=gbk
"""
检测0008是否产生对应迭代次数的相似性度量，借此分批提交任务。
1. 检测0008是否存在某个文件
2. 检测后提交脚本文件 传递参数
"""
import time
import os
from my_logger import MyLog
import sys
my_logger = MyLog().logger

pre_hour = 24
step = 5
is_transfer = int(sys.argv[1])
start_iter = int(sys.argv[2])

transfer_flag = "no_transfer" if is_transfer == 0 else "transfer"

root_dir = f"{pre_hour}h"
PSM_SAVE_PATH = f'/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/result/psm_with_xgb/{root_dir}/psm_{transfer_flag}/'

iteration_idx = start_iter

my_logger.warning(f"start checking ... [params] - is_transfer:{is_transfer}, start_iter:{start_iter}")
while iteration_idx <= 120:
    psm_file_name = os.path.join(PSM_SAVE_PATH, f"0008_{pre_hour}h_{iteration_idx}_psm_boost50_{transfer_flag}.csv")
    if os.path.exists(psm_file_name):
        my_logger.info(f"{psm_file_name} exist, start sbatch 0009_auto_sbatch_XGB.sh {is_transfer} {iteration_idx}")
        time.sleep(20)
        os.system(
            f'sh /panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/code_xgb_new/0009_auto_sbatch_XGB.sh {is_transfer} {iteration_idx} ')
        iteration_idx += step
    else:
        time.sleep(30)

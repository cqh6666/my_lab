#encoding=gbk
"""
检测0008是否产生对应迭代次数的相似性度量，借此分批提交任务。
1. 检测0008是否存在某个文件
2. 检测后提交脚本文件 传递参数
"""
import time
import os
from my_logger import MyLog

my_logger = MyLog().logger
PSM_SAVE_PATH = '/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/result/personal_model_with_xgb/24h_xgb_model/24h_no_transfer_psm/'
iteration_idx = 15
xgb_boost_num = 70
step = 5

my_logger.warning("start checking ...")
while iteration_idx <= 120:
    psm_no_transfer_file_name = os.path.join(
        PSM_SAVE_PATH, f'0008_24h_{iteration_idx}_feature_weight_localboost{xgb_boost_num}_no_transfer.csv')
    if os.path.exists(psm_no_transfer_file_name):
        my_logger.info(f"{psm_no_transfer_file_name} - exist...")
        time.sleep(20)
        os.system(
            f'sh /panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/code/0009_auto_sbatch.sh {iteration_idx} 0')
        iteration_idx += step
    else:
        time.sleep(30)

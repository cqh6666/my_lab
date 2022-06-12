#encoding=gbk
"""
���0008�Ƿ������Ӧ���������������Զ�������˷����ύ����
1. ���0008�Ƿ����ĳ���ļ�
2. �����ύ�ű��ļ� ���ݲ���
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

PSM_SAVE_PATH = f'/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/result/personal_model_with_lr/{pre_hour}h/psm_{transfer_flag}/'

iteration_idx = start_iter

my_logger.warning(f"start checking ... [params] - is_transfer:{is_transfer}, start_iter:{start_iter}")
while iteration_idx <= 120:
    psm_no_transfer_file_name = os.path.join(PSM_SAVE_PATH, f"0008_{pre_hour}h_{iteration_idx}_psm_{transfer_flag}.csv")
    if os.path.exists(psm_no_transfer_file_name):
        my_logger.info(f"{psm_no_transfer_file_name} exist, start sbatch 0009_auto_sbatch_LR.sh {is_transfer} {iteration_idx}")
        time.sleep(20)
        os.system(
            f'sh /panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/code_lr_new/0009_auto_sbatch_LR.sh {is_transfer} {iteration_idx} ')
        iteration_idx += step
    else:
        time.sleep(30)

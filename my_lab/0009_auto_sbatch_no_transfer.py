#encoding=gbk
"""
���0008�Ƿ������Ӧ���������������Զ�������˷����ύ����
1. ���0008�Ƿ����ĳ���ļ�
2. �����ύ�ű��ļ� ���ݲ���
"""
import time
import os
from my_logger import MyLog

my_logger = MyLog().logger
PSM_SAVE_PATH = '/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/result/personal_model_with_xgb/24h_xgb_model/24h_no_transfer_psm/'
iteration_idx = 5
xgb_boost_num = 70
step = 5
# 1 -> true , 0 -> false
is_transfer = 0

my_logger.warning("start checking ...")
while iteration_idx <= 120:
    psm_no_transfer_file_name = os.path.join(
        PSM_SAVE_PATH, f'0008_24h_{iteration_idx}_feature_weight_localboost{xgb_boost_num}_no_transfer.csv')
    if os.path.exists(psm_no_transfer_file_name):
        my_logger.info(f"{psm_no_transfer_file_name} - exist...")
        time.sleep(20)
        os.system(
            f'sh /panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/code/0009_auto_sbatch.sh {iteration_idx} {is_transfer}')
        iteration_idx += step
    else:
        time.sleep(30)

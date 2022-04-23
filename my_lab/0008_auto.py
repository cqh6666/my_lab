"""
check metric KL learning
"""
import time
import os

os.system('sbatch /panfs/pfs.local/work/liu/xzhang_sta/tangxizhuo/pg/0008_auto_learn_KL_use_XGB.sh 0')

step = 5
i = step

while i < 120:

    if os.path.exists(
            '/panfs/pfs.local/work/liu/xzhang_sta/tangxizhuo/pg/KL_result/0008_01_001_005-1_{}.csv'.format(
                    i)):

        time.sleep(20)
        os.system('sbatch /panfs/pfs.local/work/liu/xzhang_sta/tangxizhuo/pg/0008_auto_learn_KL_use_XGB.sh {}'.format(i))
        i = i + step
    else:
        time.sleep(30)

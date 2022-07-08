# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     0009_schedule_sbatch
   Description:   每5个小时自动提交40个任务
   Author:        cqh
   date:          2022/6/20 20:15
-------------------------------------------------
   Change Activity:
                  2022/6/20:
-------------------------------------------------
"""
__author__ = 'cqh'

import time
import os


def auto_sbatch(is_transfer, iter_idx, select):
    os.system(
        f"sh /panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/code_xgb_old/0009_auto_sbatch_XGB.sh {is_transfer} {iter_idx} {select}"
    )


def run(iter_idx):
    auto_sbatch(1, iter_idx, 5)
    auto_sbatch(1, iter_idx, 20)
    auto_sbatch(0, iter_idx, 5)
    auto_sbatch(0, iter_idx, 20)


if __name__ == '__main__':
    # 30 min 后运行
    time.sleep(60 * 30)
    iter_idx = 5
    while iter_idx <= 20:
        run(iter_idx)
        iter_idx += 5
        time.sleep(60 * 60 * 5)

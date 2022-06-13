#!/bin/bash
is_transfer=$1
start=$2
end=$3
step=5

while [ $start -le $end ]
do
  sbatch /panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/code_lr_old/0010_get_auc_result_lr_old.sh ${is_transfer} ${start}
  let start=start+$step
done
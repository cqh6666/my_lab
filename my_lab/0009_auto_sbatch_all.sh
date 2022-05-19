#!/bin/sh
step=1500
final=21613

iter=1
iter_step=1
max_iter=4
is_transfer=1

while [ $iter -le $max_iter ]
do
  start=0
  end=$step
  while [ $start -le $final ]
  do
    if [ $is_transfer -eq 1 ]; then
      sbatch /panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/code/0009_personal_XGB_metric_KL.sh ${start} ${end} ${iter}
    elif [ $is_transfer -eq 0 ]; then
      sbatch /panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/code/0009_personal_XGB_metric_KL_no_transfer.sh ${start} ${end} ${iter}
    fi
    let start=start+$step
    let end=end+$step
  done
  let iter=iter+$iter_step
done
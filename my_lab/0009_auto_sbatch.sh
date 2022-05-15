#!/bin/sh
final=21613

step=1500
iter=5
iter_step=5
max_iter=10

while (($iter <= $max_iter))
do
  start=0
  end=$step
  while [ $start -lt $final ]
  do
    sbatch /panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/code/0009_personal_XGB_metric_KL_no_transfer.sh ${start} ${end} ${iter}
    let start=start+$step
    let end=end+$step
  done
  let iter=iter+$iter_step
done
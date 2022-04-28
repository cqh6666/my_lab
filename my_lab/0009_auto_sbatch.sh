#!/bin/sh
final=21613

step=1500
iter=3
iter_step=1
max_iter=8

while (($iter < $max_iter))
do
  start=0
  end=$step
  while [ $start -lt $final ]
  do
    sbatch /panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/code/0009_personal_XGB_metric_KL.sh ${start} ${end} ${iter}
    let start=start+$step
    let end=end+$step
  done
  let iter=iter+$iter_step
done
#!/bin/sh
is_transfer=$1

iter=5
iter_step=5
max_iter=20

step=1500
final=21613

while [ $iter -le $max_iter ]
do
  start=0
  end=$step
  while [ $start -le $final ]
  do
    sbatch /panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/code_xgb_old/0009_test_KL_use_XGB.sh ${is_transfer} ${iter} ${start} ${end}
    let start=start+$step
    let end=end+$step
  done
  let iter=iter+$iter_step
done
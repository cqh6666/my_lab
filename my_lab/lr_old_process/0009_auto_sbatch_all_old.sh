#!/bin/sh
step=1500
final=21613

iter=5
iter_step=5
max_iter=15
is_transfer=0

while [ $iter -le $max_iter ]
do
  start=0
  end=$step
  while [ $start -le $final ]
  do
    sbatch /panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/code/0009_test_KL_use_LR_old.sh ${start} ${end} ${is_transfer} ${iter}
    let start=start+$step
    let end=end+$step
  done
  let iter=iter+$iter_step
done
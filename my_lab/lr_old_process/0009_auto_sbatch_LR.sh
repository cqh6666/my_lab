#!/bin/sh

is_transfer=$1
iter=0

start=0
final=3000
step=1000
end=$step

while [ $start -lt $final ]
do
  sbatch /panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/code_lr_old/0009_test_KL_use_LR_old.sh ${is_transfer} ${iter} ${start} ${end}
  let start=start+$step
  let end=end+$step
done

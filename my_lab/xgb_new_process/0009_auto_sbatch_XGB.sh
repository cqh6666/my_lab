#!/bin/sh

is_transfer=$1
iter=$2

start=0
final=21613
step=1500
end=$step

while [ $start -lt $final ]
do
  sbatch /panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/code_xgb_new/0009_test_KL_use_XGB.sh ${start} ${end} ${is_transfer} ${iter}
  let start=start+$step
  let end=end+$step
done

#!/bin/sh

is_transfer=$1
iter=$2
select=$3

start=0
final=21613
step=2220
end=$step

while [ $start -lt $final ]
do
  sbatch /panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/code_xgb_old/0009_test_KL_use_XGB.sh ${is_transfer} ${iter} ${select} ${start} ${end}
  let start=start+$step
  let end=end+$step
done

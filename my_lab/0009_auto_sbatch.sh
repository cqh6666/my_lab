#!/bin/sh

iter=$1
is_transfer=$2

start=0
final=21613
step=1500
end=$step

while [ $start -lt $final ]
do
  if [ $is_transfer -eq 1 ]; then
    sbatch /panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/code/0009_personal_XGB_metric_KL.sh ${start} ${end} ${iter}
  elif [ $is_transfer -eq 0 ]; then
    sbatch /panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/code/0009_personal_XGB_metric_KL_no_transfer.sh ${start} ${end} ${iter}
  fi
  let start=start+$step
  let end=end+$step
done

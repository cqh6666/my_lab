#!/bin/bash
is_transfer=$1
select=$2
start=$3
end=$4
step=5

while [ $start -le $end ]
do
  sbatch /panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/code_xgb_old/0010_get_auc_result.sh ${is_transfer} ${start} ${select}
  let start=start+$step
done
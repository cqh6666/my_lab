#!/bin/bash
start=5
step=5
end=21

while [ $start -lt $end ]
do
  sbatch /panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/code/0010_get_auc_result.sh ${start}
  let start=start+$step
done
#!/bin/bash
start=20
step=5
end=50
is_transfer=1

while [ $start -le $end ]
do
  sbatch /panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/code/0010_get_auc_result.sh ${start} ${is_transfer}
  let start=start+$step
done
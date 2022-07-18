#!/bin/sh
mean=$1
start=0
step=2000
final=10000
end=$step
while [ $start -le $final ]
do
  sbatch /panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/code_pca/code_xgb/S04_kth_avg_XGB.sh 0 ${mean} ${start} ${end}
  sbatch /panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/code_pca/code_xgb/S04_kth_avg_XGB.sh 1 ${mean} ${start} ${end}
  let start=start+$step
  let end=end+$step
done


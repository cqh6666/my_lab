#!/bin/sh
n_comp=$1
start=20
final=200
step=20
while [ $start -le $final ]
do
  sbatch /panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/code_pca/code_xgb/S05_kth_avg.sh 0 ${n_comp} ${start}
  sbatch /panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/code_pca/code_xgb/S05_kth_avg.sh 1 ${n_comp} ${start}
  let start=start+$step
done
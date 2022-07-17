#!/bin/sh
start=100
final=500
step=100
while [ $start -le $final ]
do
  sbatch /panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/code_pca/code_xgb/S01_random_XGB.sh ${start} 1
  sbatch /panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/code_pca/code_xgb/S01_random_XGB.sh ${start} 0
  let start=start+$step
done

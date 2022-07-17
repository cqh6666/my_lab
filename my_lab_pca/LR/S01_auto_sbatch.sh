#!/bin/sh
start=300
final=400
step=100
while [ $start -le $final ]
do
  sbatch /panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/code_pca/code_lr/S01_random_LR.sh ${start} 1
  sbatch /panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/code_pca/code_lr/S01_random_LR.sh ${start} 0
  let start=start+$step
done

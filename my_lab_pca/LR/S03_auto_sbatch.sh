#!/bin/sh
start=500
final=4000
step=500
while [ $start -le $final ]
do
  sbatch /panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/code_pca/code_lr/S03_pca_similar_LR.sh 1 ${start}
  sbatch /panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/code_pca/code_lr/S03_pca_similar_LR.sh 0 ${start}
  let start=start+$step
done

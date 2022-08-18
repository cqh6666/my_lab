#!/bin/sh
final=10000
step=2000
let start=0
let end=$step
dim=$1
while [ $start -lt $final ]
do
  sbatch /panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/code_pca/code_xgb/S05_auto_encoder_similar_XGB.sh 1 ${start} ${end} ${dim}
  sbatch /panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/code_pca/code_xgb/S05_auto_encoder_similar_XGB.sh 0 ${start} ${end} ${dim}
  let start=start+$step
  let end=end+$step
done

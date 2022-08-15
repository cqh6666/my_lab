#!/bin/sh
sleep 6h
final=10000
step=2000
let start=0
let end=$step
while [ $start -lt $final ]
do
  sbatch /panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/code_pca/code_xgb/S05_auto_encoder_similar_XGB.sh 1 ${start} ${end}
  sbatch /panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/code_pca/code_xgb/S05_auto_encoder_similar_XGB.sh 0 ${start} ${end}
  let start=start+$step
  let end=end+$step
done

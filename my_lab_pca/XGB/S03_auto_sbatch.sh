#!/bin/sh]
final=10000
step=2000

for comp in {20..100..20}
do
  let start=0
  let end=$step
  while [ $start -lt $final ]
  do
    sbatch /panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/code_pca/code_xgb/S03_pca_similar_XGB.sh 1 ${comp} ${start} ${end}
    sbatch /panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/code_pca/code_xgb/S03_pca_similar_XGB.sh 0 ${comp} ${start} ${end}
    let start=start+$step
    let end=end+$step
  done
done

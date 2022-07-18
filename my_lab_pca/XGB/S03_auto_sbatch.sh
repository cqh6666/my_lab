#!/bin/sh]
n_comp=$1
start=0
final=10000
step=2000
end=$step

while [ $start -le $final ]
do
  sbatch /panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/code_pca/code_xgb/S03_pca_similar_XGB.sh 1 ${n_comp} ${start} ${end}
  sbatch /panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/code_pca/code_xgb/S03_pca_similar_XGB.sh 0 ${n_comp} ${start} ${end}
  let start=start+$step
  let end=end+$step
done
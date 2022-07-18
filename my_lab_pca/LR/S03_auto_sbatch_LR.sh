#!/bin/sh]
start=0
final=10000
step=2000
end=$step

comp_list=(500 1000 2000 3000 4000)
for comp in ${comp_list[@]}
do
  let start=0
  let end=start+$step
  while [ $start -lt $final ]
  do
      sbatch /panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/code_pca/code_lr/S03_pca_similar_LR.sh 1 ${comp} ${start} ${end}
      sbatch /panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/code_pca/code_lr/S03_pca_similar_LR.sh 0 ${comp} ${start} ${end}
      let start=start+$step
      let end=end+$step
  done
done
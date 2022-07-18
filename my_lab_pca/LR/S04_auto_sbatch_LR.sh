#!/bin/sh
start=0
step=2000
final=10000
end=$step

mean_list=(20 50 100 200)
for mean in ${mean_list[@]}
do
  let start=0
  let end=$step
  while [ $start -lt $final ]
  do
      sbatch /panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/code_pca/code_lr/S04_kth_avg_LR.sh 1 ${mean} ${start} ${end}
      sbatch /panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/code_pca/code_lr/S04_kth_avg_LR.sh 0 ${mean} ${start} ${end}
      let start=start+$step
      let end=end+$step
  done
done
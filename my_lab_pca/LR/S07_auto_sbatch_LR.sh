#!/bin/sh
start=0
step=2000
final=10000
end=$step
iter_list=(60 70 80)
for iter_idx in ${iter_list[@]}; do
  let start=0
  let end=$step
  while [ $start -lt $final ]; do
    sbatch /panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/code_pca/code_lr/S07_test_result_psm_with_pca.sh 1 ${iter_idx} ${start} ${end}
    sbatch /panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/code_pca/code_lr/S07_test_result_psm_with_pca.sh 0 ${iter_idx} ${start} ${end}
    let start=start+$step
    let end=end+$step
  done
done

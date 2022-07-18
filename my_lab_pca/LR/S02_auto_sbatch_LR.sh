#!/bin/sh
start=0
step=2000
end=$step
final=10000
boost_arr=(50 100 200)
select_arr=(5 10 20)
for select in ${select_arr[@]}
do
  for boost in ${boost_arr[@]}
  do
    let start=0
    let end=$step
    while [ $start -le $final ]
    do
      sbatch /panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/code_pca/code_xgb/S02_no_pca_similar_LR.sh 1 ${boost} ${select} ${start} ${end}
      sbatch /panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/code_pca/code_xgb/S02_no_pca_similar_LR.sh 0 ${boost} ${select} ${start} ${end}
      let start=start+$step
      let end=end+$step
     done
  done
done
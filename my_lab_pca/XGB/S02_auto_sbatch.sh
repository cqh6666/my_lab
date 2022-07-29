#!/bin/sh
start=0
step=2000
end=$step
final=10000
boost_arr=(50)
select_arr=(10)
for select in ${select_arr[@]}
do
  for boost in ${boost_arr[@]}
  do
    let start=0
    let end=$step
    while [ $start -le $final ]
    do
      sbatch /panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/code_pca/code_xgb/S02_no_pca_similar_XGB.sh 1 ${boost} ${select} ${start} ${end}
      sbatch /panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/code_pca/code_xgb/S02_no_pca_similar_XGB.sh 0 ${boost} ${select} ${start} ${end}
      let start=start+$step
      let end=end+$step
     done
  done
done
#!/bin/sh
comp=$1
start=25
final=100
step=25
while [ $start -lt $final ]
do
  sbatch /panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/code_pca/S05_lle_test.sh ${comp} ${start}
  let start=start+$step
done

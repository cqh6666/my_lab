#!/bin/sh

start=0
step=1500
# step=30
end=${step}
final=20390
# final=30

iter=50
max_iter= 51

while (($iter < $max_iter))
do
  while (($start < $final))
  do
    sbatch /panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/code/0009_personal_XGB_metric_KL.sh ${start} ${end} ${iter}
    start=$(($start+$step))
    end=$(($end+$step))
  done
  iter=$(($iter+5))
done

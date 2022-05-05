#!/bin/sh

start=5
step=5
end=50

while (($start <= $end))
do
  sbatch /panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/code/0010_get_auc_result.py ${start}
  start=$(($start+$step))
done

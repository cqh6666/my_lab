#!/bin/sh

min_num_boost=$1
max_num_boost=$2
step=$3

cur_num_boost=$min_num_boost

while [ $cur_num_boost -le $max_num_boost ]
do
  sbatch /panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/code_xgb_old/0007_global_XGB.sh ${cur_num_boost}
  let cur_num_boost=cur_num_boost+step
done

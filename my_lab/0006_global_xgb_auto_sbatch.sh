#!/bin/sh

boost_nums=1
step=10

while ((boost_nums <= 50))
do
  sbatch /panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/code/0006_global_XGB.sh ${boost_nums}
  boost_nums=$((boost_nums+step))
done


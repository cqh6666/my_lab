#!/bin/sh
start=2010
end=2018

while [ $start -lt $end ]
do
  sbatch /panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/code/0002_analysis_lab_units_samples.sh ${start}
  let start=start+1
done
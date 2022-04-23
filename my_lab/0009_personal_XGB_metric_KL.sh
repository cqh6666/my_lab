#!/bin/bash
#SBATCH --job-name=BDAI_CQH_LAB                       # Job name
#SBATCH --partition=sixhour                      # Partition Name (Required)
#SBATCH --mail-type=BEGIN,END,FAIL               # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=2018ch@m.scnu.edu.cn              # Where to send mail	
#SBATCH --ntasks=1                               # Run a single task	
#SBATCH --cpus-per-task=20                        # Number of CPU cores per task
#SBATCH --mem-per-cpu=8gb                       # Job memory request
#SBATCH --time=0-06:00:00                        # Time limit days-hrs:min:sec
#SBATCH --output=./log/0009_personal_XGB_metric_KL_%j.log              # Standard output and error log
 
echo "Running on $SLURM_CPUS_PER_TASK cores"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

start=${1}
end=${2}
iter=${3}

source /panfs/pfs.local/software/7/install/anaconda/4.6/bin/activate /panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/env_38
python3.8 /panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/code/0009_personal_XGB_metric_KL.py ${start} ${end} ${iter}

# activate
conda activate /panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/env
conda activate /panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/env_38

conda deactivate


1. sbatch xxx.sh 提交任务
2. scancel (jobid) 取消任务
3. squeue -u xzhang_sta 查询当前任务状态


cp /panfs/pfs.local/work/liu/xzhang_sta/tangxizhuo/data/24/24h_snap1_rm1_remain_feature.csv /panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/data/24h/24h_999_remained_feature.csv
cp ../data/new_feature_map.csv /panfs/pfs.local/work/liu/xzhang_sta/tangxizhuo/data/new_feature_map.csv
conda create
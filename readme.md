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


/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/result/personal_model_with_xgb/24h_xgb_model/
/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/code
/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/result/personal_model_with_xgb/24h_xgb_model/24h_no_transfer_psm/
/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/result/personal_model_with_xgb/24h_xgb_model/24h_transfer_psm/

/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/result/personal_model_with_lr/24h/

## 删除旧日志
cd 当前目录
find . -mtime +5 -name "*.log" -exec rm -rf {} \;
find . -name "0009*.log" -exec rm -rf {} \;
## 移动日志 遍历当前目录
find ./* -maxdepth 0 -name "*.csv" -exec mv {} ./history_log/ \;

### 查看日志
- 按时间
ll -rt   降序(推荐，最后一行是最新的文件)
ll -t
- 按大小
ll -Sh


### 后台执行
nohup sh /panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/code_pca/code_xgb/S05_auto_sbatch.sh >> /panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/code_pca/code_xgb/S05_nohup.log &
nohup sh /panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/code_pca/code_lr/S05_auto_sbatch_LR.sh >> /panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/code_pca/code_lr/S05_nohup.log &

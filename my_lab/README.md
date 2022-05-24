# 0008,0009,0010使用
## 0008 训练集建模阶段
多线程程序涉及到的有 `0008_auto_learn_KL_use_XGB_mt.py`,`0008_auto_mt.py`.

- `0008_auto_learn_KL_use_XGB_mt.py` 在sixhour中运行，需要传入的第i次迭代；
- `0008_auto_mt.py` 在liu中运行，由于每一次迭代大概需要2h，所以一般一次提交任务只能3次迭代（稳妥一点则设置为2次），所以设置了个自动脚本，当检测到第i次迭代完成（**会生成相似性度量csv文件**），就提交新的迭代，从i开始迭代。

```shell script
sbatch 0008_auto_mt.sh
```
## 0009 测试集建模阶段
涉及到的有 `0009_personal_XGB_metric_KL.py`,`0009_auto_sbatch.sh`

测试集有2w+个样本，如果按顺序对每一个建模，需要消耗大量时间。由于2w+样本可以并行处理，没有顺序问题，所以同样使用多线程进行分批处理。
我们将测试集分成15份，每1500一份数据。通过`${start} ${end} ${iter}`参数使用`0009_auto_sbatch.sh`批量提交任务。
> (start,end) 代表2w+样本的索引, (iter) 代表第i次迭代次数

```shell script
sh 0009_auto_sbatch.sh
```

## 0010 计算auc。
将0009每次迭代多份数据进行合并成为1个文件，然后计算最终的auc结果；
涉及到的有 `0010_get_auc_result.py`, `0010_auto_get_auc_result.sh`
- `0010_auto_get_auc_result.sh` 也是自动脚本，通过赋予迭代参数进行处理。
```shell script
sh 0010_auto_get_auc_result.sh
```


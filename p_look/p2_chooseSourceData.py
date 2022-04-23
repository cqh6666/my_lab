import numpy as np
import pandas as pd
from sklearn import tree
import dataUtil
import pickle

class ChooseData:
    def __init__(self, data, label, N, scope):
        self.data = data
        self.label = label
        self.N = N
        self.scope = scope

    # 入口函数
    def chooseData(self):
        # 读取模型权重
        file_name = "./model/weight.pkl"
        weight_file = open(file_name, 'rb')
        weight = pickle.load(weight_file)
        weight = weight / np.sum(weight)
        # 设置一个scope数组保存源域数据得分
        scope = np.zeros((self.data.shape[0]))
        chooseData = []
        chooseLabel = []
        # 加载目标域模型,计算本次model的scope
        for i in range(self.N):
            file_name = "./model/model_" + str(i+1) + ".pkl"
            model_file = open(file_name, 'rb')
            clf = pickle.load(model_file)
            pred = clf.predict(self.data)
            error = np.sum(np.abs(pred-self.label)) / pred.shape[0]
            print("error: ", error)
            # 计算scope
            scope += np.abs(pred - self.label)*weight[i]
        # 选择源域数据
        for i in range(scope.shape[0]):
            if scope[i] <= self.scope:
                chooseData.append(self.data[i])
                chooseLabel.append(self.label[i])
        print("scope count: ", np.sum(scope<=0.3), np.sum(scope<=0.2), np.sum(scope<=0.1), np.sum(scope<=0.05))
        print("The Number of minor class: ", np.sum(chooseLabel))
        chooseData = np.array(chooseData)
        print("data shape: ", chooseData.shape)
        chooseLabel = np.array(chooseLabel)
        chooseData = np.insert(chooseData, 0, values=chooseLabel, axis=1)
        chooseData = pd.DataFrame(chooseData)
        # 保存数据
        file_name = "./data/sourceChooseData.csv"
        chooseData.to_csv(file_name, index=None)


if __name__ == '__main__':
    # AKI_1_2_3 2015 1 1 1
    task_name = "AKI_1_2_3"
    year = str(2015)
    pre_day = "24h"
    del_scr_bun = 1
    IsDel = "scr_bun" if del_scr_bun == 0 else "no_scr_bun"
    IsFS = 1
    FS = "no_FS" if IsFS == 0 else "FS"

    data_parent_path = "/panfs/pfs.local/work/liu/xzhang_sta/yaorunze/iTrRos/data/"
    # 特征路径
    all_feature_path = '/panfs/pfs.local/work/liu/xzhang_sta/yaorunze/iTrRos/data/feature/feature_name.csv'
    feature_path = f'{data_parent_path}/feature/{task_name}_{year}_{pre_day}_features2.csv'
    # 读取源域数据
    source_path = f'{data_parent_path}/source/2010_data.csv'
    print("--------source--------")
    print('source_path: ', source_path)
    data, label = dataUtil.getData(source_path, all_feature_path, feature_path, del_scr_bun)
    data = data.fillna(value=0)
    data = np.array(data)
    label = np.array(label)

    c = ChooseData(data, label, 10, 0.1)
    c.chooseData()
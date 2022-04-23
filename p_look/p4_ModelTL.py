import numpy as np
import pickle
from lightgbm import LGBMClassifier
from sklearn import tree
import dataUtil

class ModelTransfer:
    def __init__(self, data, label, N):
        self.data = data
        self.label = label
        self.N = N

    # 入口函数
    def modelTransfer(self):
        weight = []
        for i in range(self.N):
            # 1.读取模型参数
            file_name = "./bestParam/param_" + str(i+1) + ".pkl"
            best_param = open(file_name, 'rb')
            best_param = pickle.load(best_param)
            print(best_param)
            # boosting_type = best_param['boosting_type']
            # max_depth = best_param['max_depth']
            # learning_rate = best_param['learning_rate']
            # n_estimators = best_param['n_estimators']
            # scale_pos_weight = best_param['scale_pos_weight']
            max_depth = best_param['max_depth']
            criterion = best_param['criterion']
            class_weight = best_param['class_weight']
            # 2.训练并保存模型及其正确率
            # clf = LGBMClassifier(boosting_type=boosting_type, max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimators, scale_pos_weight=scale_pos_weight)
            clf = tree.DecisionTreeClassifier(random_state=22, criterion=criterion, max_depth=max_depth, class_weight=class_weight)
            clf.fit(self.data, self.label)
            file_name = "./tl_model/model_" + str(i+1) + ".pkl"
            with open(file_name, 'wb') as file:
                pickle.dump(clf, file)
            pred = clf.predict(self.data)
            error = np.sum(np.abs(pred, self.label)) / pred.shape[0]
            print(error)
            weight.append(1-error)
        with open("./tl_model/weight.pkl", 'wb') as file:
            pickle.dump(weight, file)

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

    # 读取目标域数据
    target_path = f'{data_parent_path}/target/train_data.csv'
    print("--------target--------")
    print('target_path: ', target_path)
    data, label = dataUtil.getData(target_path, all_feature_path, feature_path, del_scr_bun)
    data = data.fillna(value=0)
    data = np.array(data)
    label = np.array(label)

    model = ModelTransfer(data, label, 10)
    model.modelTransfer()
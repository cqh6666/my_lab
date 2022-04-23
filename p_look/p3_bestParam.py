import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn import tree
from sklearn.model_selection import GridSearchCV
import pickle

class BestParam:
    def __init__(self, data, label, N):
        self.data = data
        self.label = label
        self.N = N

    # 有放回抽取样本
    def sampling(self):
        index = np.random.choice(self.data.shape[0], self.data.shape[0])
        data = []
        label = []
        for i in index:
            data.append(self.data[i])
            label.append(self.label[i])
        data = np.array(data)
        label = np.array(label)
        return data, label

    # 计算模型的最优参数
    def calculate_best_param(self, data, label):
        # 设置模型
        #clf = LGBMClassifier()
        # 设置参数
        #param = {"boosting_type":['gbdt', 'goss'],
        #         "max_depth":[8, 10, 15],
        #         "learning_rate":[0.1, 0.3],
        #         'n_estimators':[80, 100, 120],
        #         'scale_pos_weight':[1, 3, 5, 6]}
        
        clf = tree.DecisionTreeClassifier(random_state=22)
        param = {"max_depth":[15, 20, 25],
                 "criterion":['gini','entropy'],
                 "class_weight":[{1:1},{1:2},{1:3},{1:5}]}
        clf = GridSearchCV(clf, param_grid=param, cv=3)
        clf.fit(data, label)
        best_param = clf.best_params_
        print(best_param)
        return best_param

    # 入口函数
    def bestParam(self):
        for i in range(self.N):
            # 1.有放回抽样
            data, label = self.sampling()
            # 2.最优参数计算
            best_param = self.calculate_best_param(data, label)
            file_name = "./bestParam/param_" + str(i+1) + ".pkl"
            with open(file_name, 'wb') as file:
                pickle.dump(best_param, file)

        return

if __name__ == '__main__':
    data = pd.read_csv("./data/sourceChooseData.csv")
    label = data.iloc[:, 0]
    data = data.iloc[:, 1:]
    data = np.array(data)
    label = np.array(label)

    bestParam = BestParam(data, label, 10)
    bestParam.bestParam()
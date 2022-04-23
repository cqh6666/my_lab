import numpy as np
import lightgbm as lgb
from sklearn import tree
import scipy.stats
import pickle
import math
import random
import dataUtil
import sys


class ITrAdaBoost:
    # 其中，Dt为目标域数据集，Ds为源域数据集，Tt为目标域标签，Ts为源域标签，test为测试样本，N为迭代次数
    def __init__(self, Dt, Ds, Tt, Ts, N, ratio, num):
        self.Dt = Dt
        self.Ds = Ds
        self.Tt = Tt
        self.Ts = Ts
        self.N = N
        self.ratio = ratio
        self.num = num

    # 训练基分类器
    def train(self, data, label, weight):
        # clf = lgb.LGBMClassifier(learning_rate=0.1, n_estimators=100)
        clf = tree.DecisionTreeClassifier(random_state=22, max_depth=20)
        clf.fit(data, label, sample_weight=weight)
        pred = clf.predict(data)
        return pred

    # 计算分类错误率
    def calculate_error_rate(self, result_label, label, weight):
        total = np.sum(weight)
        error_rate = np.sum(weight / total * np.abs(result_label - label))
        return error_rate

    # 利用模型预测概率计算js散度
    def calculate_js_distance_byproba(self, Dt, Ds, Tt, Ts):
        # 源域
        # clf = lgb.LGBMClassifier(learning_rate=0.1, n_estimators=100)
        clf = tree.DecisionTreeClassifier(random_state=22)
        Ds = np.array(Ds)
        Ts = np.array(Ts)
        clf.fit(Ds, Ts)
        source_proba = clf.predict_proba(Ds)
        # 目标域
        Dt = np.array(Dt)
        Tt = np.array(Tt)
        # clf = lgb.LGBMClassifier(learning_rate=0.1, n_estimators=100)
        clf = tree.DecisionTreeClassifier(random_state=22)
        clf.fit(Dt, Tt)
        target_proba = clf.predict_proba(Dt)
        js = []
        for i in range(source_proba.shape[0]):
            total = self.calculate_js_distance(source_proba[i], target_proba)
            js.append(total)
        js = np.array(js)
        return js

    # 计算JS散度
    def calculate_js_distance(self, p, q):
        total = 0
        for qi in q:
            M = (p + qi) / 2
            total += 0.5 * scipy.stats.entropy(p, M, base=2) + 0.5 * scipy.stats.entropy(qi, M, base=2)
        return total / q.shape[0]

    # 调整权重， 源域需要计算JS散度
    def update_weight(self, row_T, row_S, index, weights, beta_t, beta, result_label, label, data, error, js):
        len = row_T + row_S
        for j in range(len):
            # 先更新目标域样本权重
            if j < row_T:
                weights[index + 1, j] = weights[index, j] * np.power(beta_t, -np.abs(result_label[j] - label[j]))
            elif (j >= row_T and j < (row_T + row_S)):
                # 更新源域样本的权重，需要先计算JS散度
                # TODO js散度计算策略更改
                if js[j - row_T] < 0.3:
                    weights[index + 1, j] = weights[index, j] * np.power(beta, -np.abs(result_label[j] - label[j]))
                else:
                    # weights[index + 1, j] = 2*(1-error)*weights[index, j] * np.power(beta, np.abs(result_label[j]-label[j]))
                    weights[index + 1, j] = weights[index, j] * np.power(beta, np.abs(result_label[j] - label[j]))
        return weights

    # 保存1/2的模型
    def save_model(self, data, label, weight, i):
        # clf = lgb.LGBMClassifier(learning_rate=0.1, n_estimators=100)
        clf = tree.DecisionTreeClassifier(random_state=22, max_depth=20)
        clf.fit(data, label, sample_weight=weight)
        file_name = "/panfs/pfs.local/work/liu/xzhang_sta/yaorunze/itr/model_iTrRos/th" + self.num + "/model_" + str(i) + ".pkl"
        with open(file_name, 'wb') as file:
            pickle.dump(clf, file)

    # 入口函数
    def iTrAdaBoost(self):
        # 1、合并源域和目标域的训练数据集
        data = np.concatenate((self.Dt, self.Ds), axis=0)
        label = np.concatenate((self.Tt, self.Ts), axis=0)
        # 2.1 初始化样本权重
        row_T = self.Dt.shape[0]
        row_S = self.Ds.shape[0]
        # 2.2 划分出正例样本
        Dtt = []
        Dst = []
        for i in range(self.Dt.shape[0]):
            if self.Tt[i] == 1:
                Dtt.append(data[i])
        Dtt = np.array(Dtt)
        for i in range(self.Ds.shape[0]):
            if self.Ts[i] == 1:
                Dst.append(data[row_T + i])
        Dst = np.array(Dst)
        Dt_new = []
        Tt_new = []
        Ds_new = []
        Ts_new = []
        # 目标域生成新的size*ratio个样本的下标，并取得重采样样本
        size = math.ceil(Dtt.shape[0] * self.ratio)
        index_target = random.sample(range(0, Dtt.shape[0]), size)
        for k in index_target:
            Dt_new.append(Dtt[k])
            Tt_new.append(1)
        Dt_new = np.array(Dt_new)
        Tt_new = np.array(Tt_new)
        # 源域生成新的
        size = math.ceil(Dst.shape[0] * self.ratio)
        index_source = random.sample(range(0, Dst.shape[0]), size)
        for k in index_source:
            Ds_new.append(Dst[k])
            Ts_new.append(1)
        Ds_new = np.array(Ds_new)
        Ts_new = np.array(Ts_new)
        # 初始化训练数据、样本权重、保存结果的数组
        Dt = np.concatenate((self.Dt, Dt_new), axis=0)
        Ds = np.concatenate((self.Ds, Ds_new), axis=0)
        Tt = np.concatenate((self.Tt, Tt_new))
        Ts = np.concatenate((self.Ts, Ts_new))
        # 生成新的训练数据
        data = np.concatenate((Dt, Ds), axis=0)
        label = np.concatenate((Tt, Ts), axis=0)
        # 初始化样本权重
        row_T = Dt.shape[0]
        row_S = Ds.shape[0]
        weight_T = np.ones([row_T, 1]) / row_T
        weight_S = np.ones([row_S, 1]) / row_S
        weight = np.concatenate((weight_T, weight_S), axis=0)
        weights = np.ones([self.N + 1, weight.shape[0]])
        weights[0][:] = weight.T
        # 初始化结果保存集
        result_label = np.ones([self.N, row_T + row_S])
        beta = 1 / (1 + np.sqrt(2 * np.log(row_S / self.N )))
        beta_t_list = []
        count = 1
        # calculate js-value
        js = self.calculate_js_distance_byproba(Dt, Ds, Tt, Ts)
        print(js.shape, np.sum(js < 0.5), np.sum(js < 0.4), np.sum(js < 0.3), np.sum(js < 0.2), np.sum(js < 0.1), np.sum(js < 0.05))
        print("params initial finished")

        # 3、迭代训练弱分类器调整权重
        for i in range(self.N):
            # 训练弱分类器，得到所有样本的分类标签
            weight = weights[i][:].T
            result = self.train(data, label, weight)

            # 计算分类错误率
            error_rate = self.calculate_error_rate(result[:row_T], Tt, weights[i][:row_T].T)
            # print("第%d轮迭代错误率为：%.2f" %(i+1, error_rate*100) + "%")
            print('%.2f' % (error_rate * 100) + '%')
            # 错误类是否大于50%
            if error_rate > 0.5:
                i = i + 1
                print("error rate greater than 50%")
                break
            if error_rate == 0:
                # 过拟合了可能
                print("error rate is 0")
                break

            # 更新样本权重，源域需要计算JS散度
            beta_t = error_rate / (1 - error_rate)
            weights = self.update_weight(row_T, row_S, i, weights, beta_t, beta, result, label, data, error_rate, js)
            # 归一化样本权重
            # weights[i+1,:row_T] = weights[i+1,:row_T] / np.sum(weights[i+1,:row_T])
            # weights[i+1,row_T:] = weights[i+1,row_T:] / np.sum(weights[i+1,row_T:])

            # 当迭代次数大于1/2时，保存模型到本地，测试时，读取模型，得到测试数据的分类标签，投票选择最优结果
            #if i >= self.N / 2:
            if i >= 0:
                beta_t_list.append(beta_t)
                self.save_model(data, label, weight, count)
                count = count + 1

            if i == self.N - 1:
                file_name = "/panfs/pfs.local/work/liu/xzhang_sta/yaorunze/itr/model_iTrRos/th" + self.num + "/weight.pkl"
                with open(file_name, 'wb') as file:
                    pickle.dump(beta_t_list, file)


if __name__ == '__main__':
    num = str(sys.argv[1])
    # AKI_1_2_3 2015 1 1 1
    task_name = "AKI_1_2_3"
    year = str(2013)
    pre_day = "24h"
    del_scr_bun = 1
    IsDel = "scr_bun" if del_scr_bun == 0 else "no_scr_bun"
    IsFS = 1
    FS = "no_FS" if IsFS == 0 else "FS"
    print(num)

    data_parent_path = "/panfs/pfs.local/work/liu/xzhang_sta/yaorunze/iTrRos/data/"
    # 特征路径
    all_feature_path = '/panfs/pfs.local/work/liu/xzhang_sta/yaorunze/iTrRos/data/feature/feature_name.csv'
    feature_path = f'{data_parent_path}/feature/{task_name}_{year}_{pre_day}_features.csv'
    # 读取源域数据
    source_path = f'{data_parent_path}/source/2010_data2.csv'
    print("--------source--------")
    print('source_path: ', source_path)
    Ds, Ts = dataUtil.getData(source_path, all_feature_path, feature_path, del_scr_bun)
    Ds = Ds.fillna(value=0)

    # 读取目标域数据
    target_path = f'{data_parent_path}/target/2013_train_data1.csv'
    print("--------target--------")
    print('target_path: ', target_path)
    Dt, Tt = dataUtil.getData(target_path, all_feature_path, feature_path, del_scr_bun)
    Dt = Dt.fillna(value=0)

    tf = ITrAdaBoost(Dt, Ds, Tt, Ts, 20, 0.5, num)
    tf.iTrAdaBoost()
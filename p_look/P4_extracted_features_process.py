# -*- coding: utf-8 -*-#
"""
This is the code for statistical the percentage of missing features in the dataset.
The program requires two parameters:
parameters 1: The name of the data center to be processed(eg:KUMC)
parameters 2: The name of the task you need to perform(eg:AKI_1_2_3 or AKI_2_3 or AKI_3  )
parameters 3: Number of days predicted in advance (1, represents one day in advance)
parameters 4: Whether to make feature selection( 0 means no features select, and 1 means make features select )

eg:
python P4_extracted_features_process.py AKI_1_2_3 2016 1 1
"""

import sys
import os
import pandas as pd
import numpy as np
from functionUtils import get_filelist, fixed_missing_rate
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from scipy import stats

# 计算卡方分布
def chi2_FS(X, y):
    return X.iloc[:, chi2(np.abs(X), y)[1] < 0.05].columns.tolist()

# 计算所提供样本的方差分析f值。
def f_classif_FS(X, y):
    return X.iloc[:, f_classif(np.abs(X), y)[1] < 0.05].columns.tolist()


def mi(X, y):
    return X.iloc[:, mutual_info_classif(X, y)>0.005].columns.tolist()

def t_test_FS(datas, labels):
    columns = datas.columns.values
    labels = labels.values
    datas = datas.values
    feature_num = datas.shape[1]
    positive_sample_index = labels == 1
    positive_sample = datas[positive_sample_index, :]
    negative_sample_index = labels == 0
    negative_sample = datas[negative_sample_index, :]
    final_feature = []

    for i in range(feature_num):
        positive_pre_feature = positive_sample[:, i]
        negative_pre_feature = negative_sample[:, i]
        _, p = stats.ttest_ind(positive_pre_feature, negative_pre_feature, equal_var=True)
        if p < 0.05:
            final_feature.append(i)
    return columns[final_feature].tolist()


def logistic_regression_FS(datas, labels):
    columns = datas.columns.values
    feature_num = datas.shape[1]
    final_feature = []
    for i in range(feature_num):
        pre_data = datas.iloc[:, i].values.reshape(-1, 1)
        lr = LogisticRegression()
        lr.fit(pre_data, labels)
        Weight_importance = lr.coef_[0]
        predProbs = lr.predict_proba(pre_data)
        # Design matrix -- add column of 1's at the beginning of your X_train matrix
        X_design = np.hstack([np.ones((pre_data.shape[0], 1)), pre_data])
        # Initiate matrix of 0's, fill diagonal with each predicted observation's variance
        V = np.product(predProbs, axis=1)
        # Note that the @-operater does matrix multiplication in Python 3.5+, so if you're running
        # Python 3.5+, you can replace the covLogit-line below with the more readable:
        # covLogit = np.linalg.inv(X_design.T @ V @ X_design)
        covLogit = np.linalg.pinv(X_design.T * V @ X_design)
        # Standard errors
        Var = np.diag(covLogit)
        Var_feature = np.zeros(1)
        Var_feature.setflags(write=True)
        Var_feature = Var[1:]
        se = np.sqrt(Var_feature)
        if Weight_importance < 0:
            final = (Weight_importance + 1.96 * se) < 0
        else:
            final = (Weight_importance - 1.96 * se) > 0
        if final:
            final_feature.append(i)
    return columns[final_feature].tolist()


if __name__ == "__main__":
    # site_name = str(sys.argv[1])
    task_name = str(sys.argv[1])
    year = str(sys.argv[2])
    pre_day = str(int(sys.argv[3]) * 24) + "h"
    IsFS = int(sys.argv[4])
    FS = "no_FS" if IsFS == 0 else "FS"
    parent_path = '/panfs/pfs.local/work/liu/xzhang_sta/yaorunze/data/'

    # load data
    data_path = f'{parent_path}/data/{task_name}/{year}/{pre_day}/no_rolling/data.csv'
    #data_path = f'{parent_path}/data/{task_name}/{year}/{pre_day}/train/data.csv'
    # save path
    save_path = f'{parent_path}/feature'
    # Feature name file path
    feature_name_path = f'{parent_path}/feature/feature_name.csv'
    feature_names = pd.read_csv(feature_name_path)["name"].values

    data_path_list = get_filelist(data_path)

    # Read the training data from the file
    train_data = pd.concat([pd.read_csv(i) for i in data_path_list])
    label = train_data.iloc[:, -2]
    train_data = train_data.iloc[:, :-2]
    train_data = train_data.fillna(value=0)
    print("data loaded.", train_data.shape)

    # Filter missing feature, if the feature missing rate is higher than fixed_missing_rate
    sample_num = train_data.shape[0]
    feature_missing_info = pd.DataFrame(feature_names[:-2], columns=["name"])
    missing_rate = ((train_data == 0).sum(axis=0) / sample_num).values
    feature_missing_info["missing"] = missing_rate
    feature_missing_info.to_csv(save_path + "/" + task_name + "_" +  year + "_" + pre_day + '_features_missing.csv', index=False)
    del feature_missing_info
    keep_index = missing_rate < fixed_missing_rate
    train_data = train_data.iloc[:, keep_index]
    data_feature = feature_names[:-2][keep_index]
    feature_names = []
    train_data.columns = data_feature
    if IsFS == 0:
        final_feature_names = data_feature
    else:
        lab, ccs, px, med, essential_features = [], [], [], [], []
        for name in data_feature:
            if "LAB_RESULT_CM" in name:
                lab.append(name)
            elif "DIAGNOSIS" in name:
                ccs.append(name)
            elif "PROCEDURE" in name:
                px.append(name)
            elif "PRESCRIBING" in name:
                med.append(name)
            else:
                essential_features.append(name)
        print("Feature number of lab=", len(lab), "ccs=", len(ccs), "px=", len(px), "med=", len(med))
        # Select feature via chi2
        ccs_px_data = train_data[ccs + px]
        ccs, px = [], []
        # 计算卡方值
        ccs_px_feature_names = chi2_FS(ccs_px_data, label)
        
        ccs_px_feature_names = mi(train_data[ccs_px_feature_names], label)
        
        ccs_px_data = []
        print("Feature number of ccs_px =", len(ccs_px_feature_names))
        # logistic regression
        lab_med_data = train_data[lab + med]
        lab, med = [], []
        # lab_med_feature_names = logistic_regression_FS(lab_med_data, label)
        # 计算所提供样本的方差分析f值。
        lab_med_feature_names = f_classif_FS(lab_med_data, label)
        
        lab_med_feature_names = mi(train_data[lab_med_feature_names], label)
        
        # lab_med_feature_names = t_test_FS(lab_med_data, label)
        del lab_med_data
        print("Feature number of lab_med =", len(lab_med_feature_names))
        final_feature_names = essential_features + ccs_px_feature_names + lab_med_feature_names
        essential_features, ccs_px_feature_names, lab_med_feature_names = [], [], []
        final_feature_names.sort(key=list(data_feature).index)
    print("Feature number =", len(final_feature_names))
    final_features = pd.DataFrame(final_feature_names, columns=["name"])
    train_data, label, final_feature_names, data_feature = [], [], [], []
    os.makedirs(save_path, exist_ok=True)
    final_features.to_csv(save_path + "/" + task_name + "_" +  year + "_" + pre_day + '_features.csv', index=False)
    print(save_path)

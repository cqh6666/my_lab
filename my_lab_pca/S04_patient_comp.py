# -*- coding: gbk -*-
"""
-------------------------------------------------
   File Name:     S04_patient_comp
   Description:   比较pca前后患者的相似程度
   Author:        cqh
   date:          2022/7/7 12:52
-------------------------------------------------
   Change Activity:
                  2022/7/7:
-------------------------------------------------
"""
__author__ = 'cqh'

import pickle

model_name = "XGB"
result_file = f"./result/S02_test_similar_patient_ids_{model_name}.pkl"
result_file2 = f"./result/S03_test_similar_patient_ids.pkl.pkl"
result_file3 = f"./result/S02_test_similar_patient_ids.pkl.pkl"

with open(result_file, 'rb') as file:
    patient_dict = pickle.load(file)
    keys = list(patient_dict.keys())
    len_value = len(patient_dict.get(keys[0]))
    print(len(keys), keys[0:10], len_value)

print("===============================================")
with open(result_file2, 'rb') as file:
    patient_dict = pickle.load(file)
    keys = list(patient_dict.keys())
    len_value = len(patient_dict.get(keys[0]))
    print(len(keys), keys[0:10], len_value)
print("===============================================")
with open(result_file3, 'rb') as file:
    patient_dict = pickle.load(file)
    keys = list(patient_dict.keys())
    len_value = len(patient_dict.get(keys[0]))
    print(len(keys), keys[0:10], len_value)

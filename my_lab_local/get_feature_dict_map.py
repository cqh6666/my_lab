# -*- coding: gbk -*-
"""
-------------------------------------------------
   File Name:     get_feature_dict_map
   Description:   ...
   Author:        cqh
   date:          2022/5/13 17:16
-------------------------------------------------
   Change Activity:
                  2022/5/13:
-------------------------------------------------
"""
__author__ = 'cqh'
import joblib
import pandas as pd

file_path = "D:\\lab\\other_file\\feature_dict_BDAI_map.pkl"
save_path = "D:\\lab\\other_file\\old_feature_map.csv"

feature_list = joblib.load(file_path)
feature_list.insert(0, "encounter_id")

feature_dict = {'feature': feature_list}
df = pd.DataFrame(feature_dict)
df.to_csv(save_path, index=False, header=0)

data = pd.read_csv(save_path, header=None)
data_list = data.values.tolist() # [ [] [] [] [] ]
data_list_2 = data.squeeze().tolist()


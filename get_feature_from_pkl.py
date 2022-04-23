# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     get_feature_from_pkl
   Description:   ...
   Author:        cqh
   date:          2022/4/14 10:58
-------------------------------------------------
   Change Activity:
                  2022/4/14:
-------------------------------------------------
"""
__author__ = 'cqh'
import joblib

parent_path = '/panfs/pfs.local/work/liu/xzhang_sta/yaorunze/data/'
map_file_path = parent_path + "/feature/feature_dict_map.pkl"
map_data = joblib.load(map_file_path)  # dict


print("column len:",len(map_data))

day_index, aki_label_index, id_index = map_data["days"], map_data["AKI_label"], map_data["encounter_id"]

print("day_index,aku_label_index,id_index")
print(day_index,aki_label_index,id_index)
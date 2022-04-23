# -*- coding: utf-8 -*-
"""
This is the code that maps the features of the data set to the dictionary
The program requires two parameters:
parameters 1: The feature_dict.csv file path for the current center(eg:/path/.../feature_dict.csv)
parameters 2: The name of the data center to be processed(eg:KUMC)

output:"feature_dict_map.pkl" and "feature_name.csv"

Example：
python P2_mapping_process.py  /path/.../feature_dict.csv  KUMC
"""
# 处理特征的

import sys
import joblib
import pandas as pd
import os


# Read the contents of the CSV file
def mapping_process(filename, save_path):
    # SCR,BUN and SCR_BUN
    # local
    scr_bun_list = ['lab222|LAB_RESULT_CM|2160-0', 'lab580|LAB_RESULT_CM|38483-4', 'LAB_RESULT_CM|14682-9',
                    'LAB_RESULT_CM|21232-4',
                    'LAB_RESULT_CM|35203-9', 'LAB_RESULT_CM|44784-7', 'LAB_RESULT_CM|59826-8',
                    'LAB_RESULT_CM|16188-5', 'LAB_RESULT_CM|16189-3', 'LAB_RESULT_CM|59826-8', 'LAB_RESULT_CM|35591-7',
                    'LAB_RESULT_CM|50380-5', 'LAB_RESULT_CM|50381-3', 'LAB_RESULT_CM|35592-5',
                    'LAB_RESULT_CM|44784-7', 'LAB_RESULT_CM|11041-1', 'LAB_RESULT_CM|51620-3', 'LAB_RESULT_CM|72271-0',
                    'LAB_RESULT_CM|11042-9', 'LAB_RESULT_CM|51619-5', 'LAB_RESULT_CM|35203-9', 'LAB_RESULT_CM|14682-9',
                    'LAB_RESULT_CM|12966-8',
                    'LAB_RESULT_CM|12965-0', 'lab739|LAB_RESULT_CM|6299-2', 'LAB_RESULT_CM|59570-2',
                    'LAB_RESULT_CM|12964-3',
                    'LAB_RESULT_CM|49071-4', 'LAB_RESULT_CM|72270-2',
                    'LAB_RESULT_CM|11065-0', 'lab457|LAB_RESULT_CM|3094-0', 'LAB_RESULT_CM|35234-4',
                    'LAB_RESULT_CM|14937-7',
                    'LAB_RESULT_CM|3097-3', 'LAB_RESULT_CM|44734-2'
                    ]
    # multicenter
    # scr_bun_list = ['LAB_RESULT_CM|2160-0', 'LAB_RESULT_CM|38483-4', 'LAB_RESULT_CM|14682-9', 'LAB_RESULT_CM|21232-4',
    #                 'LAB_RESULT_CM|35203-9', 'LAB_RESULT_CM|44784-7', 'LAB_RESULT_CM|59826-8',
    #                 'LAB_RESULT_CM|16188-5', 'LAB_RESULT_CM|16189-3', 'LAB_RESULT_CM|59826-8', 'LAB_RESULT_CM|35591-7',
    #                 'LAB_RESULT_CM|50380-5', 'LAB_RESULT_CM|50381-3', 'LAB_RESULT_CM|35592-5',
    #                 'LAB_RESULT_CM|44784-7', 'LAB_RESULT_CM|11041-1', 'LAB_RESULT_CM|51620-3', 'LAB_RESULT_CM|72271-0',
    #                 'LAB_RESULT_CM|11042-9', 'LAB_RESULT_CM|51619-5', 'LAB_RESULT_CM|35203-9', 'LAB_RESULT_CM|14682-9',
    #                 'LAB_RESULT_CM|12966-8',
    #                 'LAB_RESULT_CM|12965-0', 'LAB_RESULT_CM|6299-2', 'LAB_RESULT_CM|59570-2', 'LAB_RESULT_CM|12964-3',
    #                 'LAB_RESULT_CM|49071-4', 'LAB_RESULT_CM|72270-2',
    #                 'LAB_RESULT_CM|11065-0', 'LAB_RESULT_CM|3094-0', 'LAB_RESULT_CM|35234-4', 'LAB_RESULT_CM|14937-7',
    #                 'LAB_RESULT_CM|3097-3', 'LAB_RESULT_CM|44734-2'
    #                 ]
    demo_vital_index_str = \
        "demo1,demo2,demo2,demo2,demo2,demo2,demo2,demo3,demo3,demo3,demo3,demo3,demo3,demo3,demo3,demo3,demo3,demo4," \
        "demo4,demo4,demo4,demo4,demo4,vital1,vital2,vital3,vital4,vital4,vital4,vital4,vital4,vital4,vital4,vital4," \
        "vital4,vital4,vital4,vital5,vital5,vital5,vital5,vital5,vital5,vital5,vital5,vital6,vital6,vital6,vital6," \
        "vital6,vital6,vital6,vital6,vital7,vital8"
    demo_vital_index = demo_vital_index_str.split(",")
    feature_dict_map = {}
    feature_name = []
    data = pd.read_csv(filename)
    index = data["VAR_IDX"]
    table_names = data["TABLE_NAME"]
    value = data["VALUESET_ITEM"] # 对应属性的取值
    for i in range(len(demo_vital_index)):
        var_idx = demo_vital_index[i]
        valueSet_item = value[i].split('\'')[1] #取到当前属性的取值
        key = var_idx + "|" + valueSet_item
        # Adds data set features to the collection
        feature_name.append(key)
        if "demo2" == var_idx or "demo3" == var_idx or "demo4" == var_idx or "vital4" == var_idx \
                or "vital5" == var_idx or "vital6" == var_idx:
            # Add the features of the data set involved in data preprocessing to the dictionary
            feature_dict_map[var_idx + valueSet_item] = i
        else:
            # Add the features of the data set involved in data preprocessing to the dictionary
            feature_dict_map[var_idx] = i

    feature_index = len(demo_vital_index)
    lab_num = 0
    ccs_px_num = 0
    med_num = 0
    for r in range(len(demo_vital_index), len(index)):
        var_idx = index[r]
        table_name = table_names[r]
        valueSet_item = value[r].split('\'')[1]
        # The key used to run locally
        key = var_idx + "|" + table_name + "|" + valueSet_item
        # Key for multicenter operation
        # key = table_name + "|" + valueSet_item
        # Adds data set features to the collection
        feature_name.append(key)
        feature_dict_map[var_idx] = feature_index
        feature_index += 1
        # 这里为什么要对scr_bun_list里面的属性多拼接一次"|change"，再放到集合中
        if key in scr_bun_list:
            feature_name.append(key + "|change")
            feature_dict_map[var_idx + "|change"] = feature_index
            feature_index += 1

    # days:Predict the time
    # AKI_label:The class label of the sample
    # encounter_id: The sample id
    new_feature = ['bp_slp', 'bp_min', 'days', 'AKI_label', 'encounter_id']
    for feature in new_feature:
        feature_dict_map[feature] = feature_index
        feature_name.append(feature)
        feature_index += 1
    for name in feature_name:
        if "LAB_RESULT_CM" in name:
            lab_num += 1
        elif "PRESCRIBING" in name:
            med_num += 1
        elif "DIAGNOSIS" in name:
            ccs_px_num += 1
        elif "PROCEDURE" in name:
            ccs_px_num += 1
        else:
            pass
    demo_vital_num = len(demo_vital_index)
    new_feature_num = len(new_feature)
    list_num = [demo_vital_num, lab_num, ccs_px_num, med_num, new_feature_num]
    joblib.dump(feature_dict_map, save_path + "/feature_dict_map" + '.pkl')
    feature_name = pd.DataFrame(feature_name, columns=["name"])
    feature_name.to_csv(save_path + "/feature_name.csv", index=False)
    joblib.dump(list_num, save_path + "/feature_num" + '.pkl')
    # 保存了一份特征名称及其映射的


if __name__ == "__main__":
    # 当前路径下
    parent_path = '/panfs/pfs.local/work/liu/xzhang_sta/yaorunze/data'
    # The feature_dict.csv file path for the current center
    # 当前中心的feature_dict.csv文件，特征文件
    feature_dict_path = '/panfs/pfs.local/work/liu/xzhang_sta/yaorunze/data/code/feature_dict.csv'
    # The name of the data center to be processed
    # 处理的数据中心的名称
    # site_name = str(sys.argv[2])

    # Gets the data store path
    # 保存的路径
    save_path = parent_path + "/feature"
    # 创建目录
    os.makedirs(save_path, exist_ok=True)

    # Read the contents of the Excel file, then generate "feature_dict_map.pkl" and "feature_name.csv"
    # 处理特征csv文件，生成"feature_dict_map.pkl" and "feature_name.csv"
    mapping_process(feature_dict_path, save_path)

# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     0002_get_pkl_to_analysis
   Description:   ...
   Author:        cqh
   date:          2022/5/12 19:14
-------------------------------------------------
   Change Activity:
                  2022/5/12:
-------------------------------------------------
"""
__author__ = 'cqh'
import joblib
import os


def get_result_by_pkl(pkl_path):
    """字典格式"""
    return joblib.load(pkl_path)


if __name__ == '__main__':
    BASE_PATH = "D:\\lab\\other_file\\"

    result_dict_list = []
    for year in range(2010,2018):
        file_path = os.path.join(BASE_PATH, f"lab_unit_result_{year}.pkl")
        result_dict_list.append(get_result_by_pkl(file_path))



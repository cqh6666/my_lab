# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:get_data_from_feather
   Description:111
   Author:cqh
   date:2022/4/13 11:26
-------------------------------------------------
   Change Activity:
                   2022/4/13:
-------------------------------------------------
"""
__author__ = 'cqh'

import os
import pandas as pd


def run():
    first_year = 2010

    for year in range(first_year, first_year + 9):
        # feather source file path
        source_file_path = f"/panfs/pfs.local/work/liu/xzhang_sta/tangxizhuo/data/24/{year}_24h_list2dataframe.feather"
        print("===========================", year, "================================")

        get_file_size(year, source_file_path)
        df = pd.read_feather(source_file_path)
        print("[head]:", df.head())
        print("[dtpes]:", df.dtypes)
        print("[size]:", df.size)
        print("[shape]:", df.shape)
        print("=========================== end ================================")


def get_file_size(year, file_path):
    fsize = os.path.getsize(file_path) / float(1024 * 1024)
    print(year, fsize, "MB.")
    return round(fsize, 2)


if __name__ == '__main__':
    run()

# -*- coding: utf-8 -*-
"""
Process the TXT format data into structured data (list object).
The program requires two parameters:
parameters 1: The folder path of TXT format data (one TXT file for each year) of the current site (eg. /path-to-dir/KUMC-TXT-data/ )
parameters 2: The site_name of the current site (eg. KUMC)

output: multiple "string2list.pkl" files prefixed by year, saved into ../resutl/site_name/data/

Example:
    python P1_data_string_to_list_process.py  /path-to-dir/KUMC-TXT-data/  KUMC
"""

import os
import sys
import joblib
from functionUtils import get_filelist


# Structure the demo class feature data
def demo_process(data):
    return data.split('_')


# Structure the vital class feature data
def vitals_process(data):
    """
    _ ; ,
    :param data:
    :return:
    """
    first_split = data.split('_')
    second_split = [first_split[i].split(';') for i in range(len(first_split))]
    third_split = [[second_split[i][j].split(',') for j in range(len(second_split[i]))] for i in
                   range(len(second_split))]
    return third_split


# Structure the ccs and px class feature data
def css_px_process(data):
    first_split = data.split('_')
    second_split = [first_split[i].split(':') for i in range(len(first_split))]
    third_split = [[second_split[i][j].split(',') for j in range(len(second_split[i]))] for i in
                   range(len(second_split))]
    return third_split


# Structure the lab and med class feature data
def lab_med_process(data):
    first_split = data.split('_')
    second_split = [first_split[i].split(':') for i in range(len(first_split))]
    third_split = [[second_split[i][j].split(';') for j in range(len(second_split[i]))] for i in
                   range(len(second_split))]
    fourth_split = [
        [[third_split[i][j][k].split(',') for k in range(len(third_split[i][j]))] for j in range(len(third_split[i]))]
        for i in range(len(third_split))]
    return fourth_split


# Structure the label class feature data
def label_process(data):
    first_split = data.split('_')
    second_split = [first_split[i].split(',') for i in range(len(first_split))]
    return second_split


if __name__ == "__main__":
    parent_path = "/panfs/pfs.local/work/liu/xzhang_sta/yaorunze/data/"
    # The folder path of TXT format data
    txt_file_path = r'/panfs/pfs.local/work/liu/xzhang_sta/public/AKI_CDM_byYear/';
    # The site site_name
    # site_name = 'KUMC'
    # Gets the data store path
    save_file_path = parent_path + "/list/"
    os.makedirs(save_file_path, exist_ok=True)
    print(save_file_path)
    # Get the TXT file under the folder
    txt_file_list = get_filelist(txt_file_path)
    print("start")

    # The center TXT format of the data in the folder path
    year = 2010
    for txt_file in txt_file_list:
        X = []
        print("txt_file:", txt_file)
        if not txt_file.endswith('.txt'):
            continue

        with open(txt_file, 'r') as f:
            for line in f.readlines():
                encounter_id = line.strip().split('"')[1]
                demo, vital, lab, ccs, px, med, label = line.strip().split('"')[3].split('|')

                demo = demo_process(demo)
                vital = vitals_process(vital)
                lab = lab_med_process(lab)
                med = lab_med_process(med)
                ccs = css_px_process(ccs)
                px = css_px_process(px)
                label = label_process(label)

                X.append([encounter_id, demo, vital, lab, ccs, px, med, label])

        # save data
        joblib.dump(X, save_file_path + "/" + str(year) + '_string2list.pkl')
        year += 1
    print("OK")
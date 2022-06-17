# -*- coding: gbk -*-
"""
-------------------------------------------------
   File Name:     0008_add_psm_feature_name
   Description:   ...
   Author:        cqh
   date:          2022/5/23 16:20
-------------------------------------------------
   Change Activity:
                  2022/5/23:
-------------------------------------------------
"""
__author__ = 'cqh'

import os
import sys
import pandas as pd


def get_feature_dict_describe_info(re_feature_list, feature_dict_df):
    """
    得到每个特征的解释
    :param re_feature_list: 筛选后的特征集
    :param feature_dict_df: 特征字典
    :return: columns = [ psm_std, 字典解释属性 ], index = [ re_feature_list ]
    """
    feature_columns = ['psm_std']
    feature_columns.extend(feature_dict_df.columns)
    feature_describe_df = pd.DataFrame(index=re_feature_list, columns=feature_columns)
    # 只对 LAB MED PX CCS进行增加解释，其余不需要
    for re_feature in re_feature_list:
        if re_feature.startswith('LAB') or re_feature.startswith('MED') or re_feature.startswith('PX') or re_feature.startswith('CCS'):
            re_feature_str = re_feature.replace('_', '').lower()
            try:
                feature_describe_df.iloc[lambda x: x.index == re_feature, 1:] = feature_dict_df.loc[
                    lambda x: x['VAR_IDX'] == re_feature_str].values
            except Exception as err:
                continue

    return feature_describe_df


def run():

    # 特征字典
    feature_dict_df = pd.read_csv(feature_dict_file)
    remained_feature_list = pd.read_csv(remained_feature_file, header=None).squeeze().tolist()[1:-1]
    feature_describe_df = get_feature_dict_describe_info(remained_feature_list, feature_dict_df)

    if iter_idx == 0:
        psm_iter = pd.read_csv(init_psm_file)
    else:
        psm_iter = pd.read_csv(iter_psm_file)
    psm_iter.index = remained_feature_list
    data_std = train_x.std().tolist()
    feature_describe_df['psm_std'] = psm_iter.iloc[:, 0].mul(data_std)

    desc_file = os.path.join(TEMP_RESULT_PATH, f"0008_{pre_hour}h_{iter_idx}_{transfer_flag}_desc.csv")
    feature_describe_df.to_csv(desc_file)
    print(f"{desc_file} - save success!")


if __name__ == '__main__':

    is_transfer = int(sys.argv[1])
    iter_idx = int(sys.argv[2])

    transfer_flag = "transfer" if is_transfer == 1 else "no_transfer"

    pre_hour = 24
    root_dir = f"{pre_hour}h_old2"
    DATA_SOURCE_PATH = f"/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/data/{root_dir}/"  # 训练集的X和Y
    MODEL_SAVE_PATH = f'/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/result/personal_model_with_lr/{root_dir}/global_model/'
    PSM_SAVE_PATH = f'/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/result/personal_model_with_lr/{root_dir}/psm_{transfer_flag}/'
    TEMP_RESULT_PATH = f"/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/result/personal_model_with_lr/{root_dir}/temp_result/"

    train_x = pd.read_feather(
        os.path.join(DATA_SOURCE_PATH, f"all_x_train_{pre_hour}_df_rm1_norm1.feather"))
    feature_dict_file = f"/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/data/feature_dict.csv"
    remained_feature_file = os.path.join(DATA_SOURCE_PATH, "remained_new_feature_map.csv")
    # init psm
    init_psm_file = os.path.join(MODEL_SAVE_PATH, f"0006_{pre_hour}h_global_lr_liblinear_400.csv")
    # iter psm
    iter_psm_file = os.path.join(PSM_SAVE_PATH, f"0008_{pre_hour}h_{iter_idx}_psm_{transfer_flag}.csv")

    run()


















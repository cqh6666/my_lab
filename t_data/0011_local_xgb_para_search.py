"""
select some sample from all data (eg: 12369, 12269)
build normal xgb, and search best parameters

input:
    argv 1: seed
    argv 2: para range id
    argv 3: the seed whether represent range

output:
    0011_search_para_seed{seed}_para{para_id}.csv

usage: < 20G
"""
import time
import sys

import numpy as np
import pandas as pd
import xgboost as xgb


def get_param_range_dict(idx):
    """maintain a dict for all kinds of parameters"""

    res = {
       # 0 always for demo test
       0: {'max_depth': (3, 4, 1), 'min_child': (1, 2, 1), 'colsample': (1, 1, 1),
           'eta': (0.1, 0.1, 1), 'iteration': (10, 11, 1)},

       1: {'max_depth': (3, 12, 2), 'min_child': (1, 10, 2), 'colsample': (0.1, 1, 3),
           'eta': (0.01, 0.1, 3), 'iteration': (10, 11, 1)},

       2: {'max_depth': (3, 13, 1), 'min_child': (1, 11, 1), 'colsample': (0.1, 1, 10, 1),
           'eta': (0.1, 0.1, 1, 2), 'iteration': (10, 11, 1)},

       3: {'max_depth': (9, 12, 1), 'min_child': (6, 9, 1), 'colsample': (0.1, 1, 10, 1),
           'eta': (0.1, 0.1, 1, 2), 'iteration': (10, 101, 10)},
       3.1: {'max_depth': (9, 12, 1), 'min_child': (6, 9, 1), 'colsample': (0.1, 1, 10, 1),
             'eta': (0.1, 0.1, 1, 2), 'iteration': (10, 41, 10)},
       3.2: {'max_depth': (9, 12, 1), 'min_child': (6, 9, 1), 'colsample': (0.1, 1, 10, 1),
             'eta': (0.1, 0.1, 1, 2), 'iteration': (50, 81, 10)},
       3.3: {'max_depth': (9, 12, 1), 'min_child': (6, 9, 1), 'colsample': (0.1, 1, 10, 1),
             'eta': (0.1, 0.1, 1, 2), 'iteration': (90, 101, 10)},

       4: {'max_depth': (9, 10, 1), 'min_child': (7, 8, 1), 'colsample': (0.7, 0.7, 1, 1),
           'eta': (0.1, 0.2, 2, 2), 'iteration': (50, 501, 50)},

       5: {'max_depth': (9, 10, 1), 'min_child': (7, 8, 1), 'colsample': (0.7, 0.7, 1, 1),
           'eta': (0.01, 0.1, 10, 2), 'iteration': (50, 51, 1)},

       6: {'max_depth': (9, 10, 1), 'min_child': (7, 8, 1), 'colsample': (0.7, 0.7, 1, 1),
           'eta': (0.11, 0.2, 10, 2), 'iteration': (10, 51, 10)},

       7: {'max_depth': (9, 10, 1), 'min_child': (7, 8, 1), 'colsample': (0.7, 0.7, 1, 1),
           'eta': (0.01, 0.01, 1, 2), 'iteration': (50, 501, 50)},

       8.1: {'max_depth': (9, 10, 1), 'min_child': (7, 8, 1), 'colsample': (0.7, 0.7, 1, 1),
             'eta': (0.02, 0.05, 4, 2), 'iteration': (100, 501, 100)},
       8.2: {'max_depth': (9, 10, 1), 'min_child': (7, 8, 1), 'colsample': (0.7, 0.7, 1, 1),
             'eta': (0.06, 0.09, 4, 2), 'iteration': (100, 501, 100)},

       9: {'max_depth': (9, 10, 1), 'min_child': (7, 8, 1), 'colsample': (0.7, 0.7, 1, 1),
           'eta': (0.01, 0.2, 20, 2), 'iteration': (100, 301, 100)},
       9.1: {'max_depth': (9, 10, 1), 'min_child': (7, 8, 1), 'colsample': (0.7, 0.7, 1, 1),
             'eta': (0.01, 0.2, 20, 2), 'iteration': (100, 101, 1)},
       9.2: {'max_depth': (9, 10, 1), 'min_child': (7, 8, 1), 'colsample': (0.7, 0.7, 1, 1),
             'eta': (0.01, 0.2, 20, 2), 'iteration': (200, 201, 1)},
       9.3: {'max_depth': (9, 10, 1), 'min_child': (7, 8, 1), 'colsample': (0.7, 0.7, 1, 1),
             'eta': (0.01, 0.2, 20, 2), 'iteration': (300, 301, 1)},

       10: {'max_depth': (9, 10, 1), 'min_child': (7, 8, 1), 'colsample': (0.7, 0.7, 1, 1),
            'eta': (0.08, 0.15, 8, 2), 'iteration': (50, 51, 1)},
       11: {'max_depth': (9, 10, 1), 'min_child': (7, 8, 1), 'colsample': (0.7, 0.7, 1, 1),
            'eta': (0.06, 0.1, 5, 2), 'iteration': (100, 101, 1)},
       12: {'max_depth': (9, 10, 1), 'min_child': (7, 8, 1), 'colsample': (0.7, 0.7, 1, 1),
            'eta': (0.03, 0.07, 5, 2), 'iteration': (200, 201, 1)},
       13: {'max_depth': (9, 10, 1), 'min_child': (7, 8, 1), 'colsample': (0.7, 0.7, 1, 1),
            'eta': (0.02, 0.06, 5, 2), 'iteration': (300, 301, 1)},

       14.1: {'max_depth': (9, 12, 1), 'min_child': (7, 8, 1), 'colsample': (0.7, 0.7, 1, 1),
              'eta': (0.01, 0.2, 20, 2), 'iteration': (100, 301, 100)},
       14.2: {'max_depth': (9, 12, 1), 'min_child': (7, 8, 1), 'colsample': (0.7, 0.7, 1, 1),
              'eta': (0.01, 0.2, 20, 2), 'iteration': (50, 51, 1)},

       15: {'max_depth': (10, 11, 1), 'min_child': (7, 8, 1), 'colsample': (0.7, 0.7, 1, 1),
            'eta': (0.1, 0.17, 8, 2), 'iteration': (50, 51, 1)},
       16: {'max_depth': (10, 11, 1), 'min_child': (7, 8, 1), 'colsample': (0.7, 0.7, 1, 1),
            'eta': (0.04, 0.08, 5, 2), 'iteration': (100, 101, 1)},
       17: {'max_depth': (10, 11, 1), 'min_child': (7, 8, 1), 'colsample': (0.7, 0.7, 1, 1),
            'eta': (0.02, 0.06, 5, 2), 'iteration': (200, 201, 1)},
       18: {'max_depth': (10, 11, 1), 'min_child': (7, 8, 1), 'colsample': (0.7, 0.7, 1, 1),
            'eta': (0.02, 0.06, 5, 2), 'iteration': (300, 301, 1)},
       19: {'max_depth': (11, 12, 1), 'min_child': (7, 8, 1), 'colsample': (0.7, 0.7, 1, 1),
            'eta': (0.1, 0.17, 8, 2), 'iteration': (50, 51, 1)},
       20: {'max_depth': (11, 12, 1), 'min_child': (7, 8, 1), 'colsample': (0.7, 0.7, 1, 1),
            'eta': (0.06, 0.1, 5, 2), 'iteration': (100, 101, 1)},
       21: {'max_depth': (11, 12, 1), 'min_child': (7, 8, 1), 'colsample': (0.7, 0.7, 1, 1),
            'eta': (0.02, 0.06, 5, 2), 'iteration': (200, 201, 1)},
       22: {'max_depth': (11, 12, 1), 'min_child': (7, 8, 1), 'colsample': (0.7, 0.7, 1, 1),
            'eta': (0.02, 0.06, 5, 2), 'iteration': (300, 301, 1)},
    }

    return res[idx]


def get_seed_range_dict(idx):
    """maintain a dict to save seed range"""

    res = {
        # 0 for demo test
        0: (0, 2, 1),

        1: (1, 11, 1),
        1.1: (1, 6, 1), 1.2: (6, 11, 1),
        1.11: (1, 3, 1), 1.12: (3, 5, 1),
        1.21: (5, 7, 1), 1.22: (7, 9, 1), 1.23: (9, 11, 1),

        2: (11, 21, 1),

        3: (21, 71, 1),
        3.1: (21, 46, 1), 3.2: (46, 71, 1),
    }

    return res[idx]


def get_init_result():
    return pd.DataFrame(columns=['auc', 'max_depth', 'min_child_weight', 'colsample_bytree', 'eta', 'iteration', 'sample_seed', 'time'])


def get_decimal_range_list(start, end, n, decimal=1):
    res = np.linspace(start, end, n)
    res = np.around(res, decimals=decimal)
    res = res.tolist()

    return res


def get_seed_range(idx):
    """return: a list including seeds"""
    # if not use seed range, return the idx as a list directly
    global if_seeds
    if if_seeds == 0:
        res = list()
        res.append(idx)
        return res

    seed_range = get_seed_range_dict(idx)
    res = eval(f"list(range{seed_range})")

    return res


def get_param_range(idx):
    para_range = get_param_range_dict(idx)
    max_depth = eval(f"list(range{para_range['max_depth']})")
    min_child_weight = eval(f"list(range{para_range['min_child']})")
    colsample_bytree = eval(f"{get_decimal_range_list.__name__}{para_range['colsample']}")
    eta = eval(f"{get_decimal_range_list.__name__}{para_range['eta']}")
    iteration = eval(f"list(range{para_range['iteration']})")

    return max_depth, min_child_weight, colsample_bytree, eta, iteration


def get_base_para():
    """get fixed parameters"""
    res = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'nthread': 20,
        'verbosity': 0,
        'eval_metric': 'logloss',
        'subsample': 1,
        'tree_method': 'hist',
        'seed': 1001,
    }

    return res


def calculate_time(start, end, decimal=3):
    """the consuming"""
    res = end - start
    res = round(res, decimal)

    return res


def get_all_sample(folder, file_name):
    file_path = f'{folder}{file_name}'

    # read feather as dataframe
    data = pd.read_feather(file_path)

    # remove the feature: 'ID'
    data.drop(['ID'], axis=1, inplace=True)

    return data


def get_dmatrix(data):
    samples_y = data['Label']
    data.drop(['Label'], axis=1, inplace=True)

    # get DMatrix data
    res = xgb.DMatrix(data, label=samples_y)

    return res


def xgb_para_grid_search(save_file):

    # init result
    res = get_init_result()

    # set xgb para range by para_id
    global para_id
    r_max_depth, r_min_child, r_colsample, r_eta, r_iteration = get_param_range(para_id)

    # get seeds range by seed_id
    global seed_id
    # r_seed is a list
    r_seed = get_seed_range(seed_id)

    # get selected_index by seeds
    # 'selected_index' is a dict: seed -> list of index
    indexes_by_seeds_file = '0011_24h_indexes_seeds_0_100.npy'
    selected_index = np.load(indexes_by_seeds_file, allow_pickle=True).item()

    # get base xgb para(fixed)
    para = get_base_para()

    # get all data
    all_data = get_all_sample(folder='/panfs/pfs.local/work/liu/xzhang_sta/tangxizhuo/data/24/',
                              file_name='all_24h_dataframe_999_feature_normalize.feather')

    # init counter
    count = 0

    # grid search
    for max_depth in r_max_depth:
        for min_child in r_min_child:
            for colsample in r_colsample:
                for eta in r_eta:
                    for iteration in r_iteration:
                        for seed in r_seed:
                            xgb_data = get_dmatrix(all_data.iloc[selected_index[seed], :].copy())
                            # add para to base para in current search
                            para['max_depth'] = max_depth
                            para['min_child_weight'] = min_child
                            para['colsample_bytree'] = colsample
                            para['eta'] = eta

                            start_time = time.time()
                            # train xgb cv
                            xgb_cv = xgb.cv(params=para,
                                            dtrain=xgb_data,
                                            num_boost_round=iteration,
                                            metrics='auc',
                                            nfold=5,
                                            early_stopping_rounds=5,
                                            maximize=True,
                                            as_pandas=False)
                            end_time = time.time()
                            # count training time
                            consume_time = calculate_time(start_time, end_time, decimal=3)

                            # get auc
                            auc = xgb_cv['test-auc-mean'][-1]

                            # save res
                            res.loc[count] = (auc, max_depth, min_child, colsample, eta, iteration, seed, consume_time)

                            # self add
                            count += 1

    # dataframe to csv
    res.to_csv(save_file, index=False)


# accept argv
# sample seed
seed_id = float(sys.argv[1]) if '.' in sys.argv[1] else int(sys.argv[1])
# search range id
# para_id = int(sys.argv[2])
para_id = float(sys.argv[2]) if '.' in sys.argv[2] else int(sys.argv[2])
# where use seeds range: 1 for use, 0 for not use
if_seeds = int(sys.argv[3])

# use seed range
if if_seeds == 1:
    # name rule: 0011_search_para_{seeds}_{para id}.csv
    save_file_name = f'0011_search_para_seeds{seed_id}_para{para_id}.csv'
# not use seed range
else:
    # name rule: 0011_search_para_{seed}_{para id}.csv
    save_file_name = f'0011_search_para_seed{seed_id}_para{para_id}.csv'

# grid search para and save csv
xgb_para_grid_search(save_file=save_file_name)

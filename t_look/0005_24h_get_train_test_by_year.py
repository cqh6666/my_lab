"""
input:
    {year}_24h_dataframe_999_feature_normalize.feather
output:
    24h_train_x_by_year.feather
    24h_train_y_by_year.feather
    24h_test_x_by_year.feather
    24h_test_y_by_year.feather
"""
import pandas as pd

# ----- get train_x and train_y -----
# init train_data
train_data = pd.DataFrame()
for year in range(2010, 2016 + 1):
    cur_data = pd.read_feather(f'/panfs/pfs.local/work/liu/xzhang_sta/tangxizhuo/data/{year}_24h_dataframe_999_feature_normalize.feather')
    train_data = pd.concat([train_data, cur_data], axis=0)

# reset index
train_data.reset_index(drop=True, inplace=True)

# get train_y (pd.Series)
train_y = train_data['Label']

# get train_x (pd.Dataframe)
train_x = train_data.drop(['ID', 'Label'], axis=1)

print("train_x shape:", train_x.shape)
print("train_y shape:", train_y.shape)

# save train_x and train_y
train_x.to_feather('/panfs/pfs.local/work/liu/xzhang_sta/tangxizhuo/data/24h_train_x_by_year.feather')
train_y.to_frame(name='Label').to_feather('/panfs/pfs.local/work/liu/xzhang_sta/tangxizhuo/data/24h_train_y_by_year.feather')


# ----- get test_x and test_y -----
# init test_data
test_data = pd.DataFrame()
for year in range(2017, 2018 + 1):
    cur_data = pd.read_feather(f'/panfs/pfs.local/work/liu/xzhang_sta/tangxizhuo/data/{year}_24h_dataframe_999_feature_normalize.feather')
    test_data = pd.concat([test_data, cur_data], axis=0)

# reset index
test_data.reset_index(drop=True, inplace=True)

# get test_y (pd.Series)
test_y = test_data['Label']

# get test_x (pd.Dataframe)
test_x = test_data.drop(['ID', 'Label'], axis=1)

print("test_x shape:", test_x.shape)
print("test_y shape:", test_y.shape)

# save test_x and test_y
test_x.to_feather('/panfs/pfs.local/work/liu/xzhang_sta/tangxizhuo/data/24h_test_x_by_year.feather')
test_y.to_frame(name='Label').to_feather('/panfs/pfs.local/work/liu/xzhang_sta/tangxizhuo/data/24h_test_y_by_year.feather')

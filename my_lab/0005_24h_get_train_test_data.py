#encoding=gbk
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
from sklearn.model_selection import train_test_split

SOURCE_FILE_PATH = '/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/data/24h/all_24h_dataframe_999_feature_normalize.feather'


def get_data_from_feather_to_save():
    all_samples = pd.read_feather(SOURCE_FILE_PATH)

    all_samples_y = all_samples['Label']
    all_samples_x = all_samples.drop(['ID', 'Label'], axis=1)

    x_train, x_test, y_train, y_test = train_test_split(all_samples_x, all_samples_y, test_size=0.15)

    print("-------------- result ---------------")
    print("x_train shape:", x_train.shape)
    print("x_test shape:", x_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)
    print("--------------  end  ----------------")

    # save feather
    save_dataFrame_to_feather(x_train, y_train, "train")
    save_dataFrame_to_feather(x_test, y_test, "test")

    print("save to feather success!")


def save_dataFrame_to_feather(x_data, y_data, file_name):
    """
    将训练集X,Y保存为feather
    :param x_data:
    :param y_data:
    :param file_name:
    :return:
    """
    x_data.reset_index(drop=True, inplace=True)
    y_data.reset_index(drop=True, inplace=True)
    x_data.to_feather(
        f'/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/data/24h/24h_all_999_normalize_{file_name}_x_data.feather')
    y_data.to_frame(name='Label').to_feather(
        f'/panfs/pfs.local/work/liu/xzhang_sta/chenqinhai/data/24h/24h_all_999_normallize_{file_name}_y_data.feather')


if __name__ == '__main__':
    get_data_from_feather_to_save()
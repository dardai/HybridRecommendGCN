import pandas as pd

def get_ml_100k():
    # 数据加载
    train_data = pd.read_csv('./ml-100k/ua.base', sep='\t', header=None,
                             names=['user_id', 'item_id', 'rating', 'timestamp'])
    test_data = pd.read_csv('./ml-100k/ua.test', sep='\t', header=None,
                             names=['user_id', 'item_id', 'rating', 'timestamp'])
    user_data = pd.read_csv('./ml-100k/u.user', sep='|', header=None, encoding='latin1')
    item_data = pd.read_csv('./ml-100k/u.item', sep='|', header=None, encoding='latin1')

    # 测试集和训练集的用户项目不同，根据训练集对测试集进行精简
    test_data = test_data[test_data['user_id'].isin(train_data['user_id']) &
                          test_data['item_id'].isin(train_data['item_id'])]

    return train_data, test_data, user_data, item_data

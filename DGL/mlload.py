import pandas as pd
from utils.databaseIo import DatabaseIo
from globalConst import DataBaseOperateType, DataBaseQuery

def get_ml_100k():
    # 数据加载
    train_data = pd.read_csv('DGL/ml-100k/ua.base', sep='\t', header=None,
                             names=['user_id', 'item_id', 'rating', 'timestamp'])
    test_data = pd.read_csv('DGL/ml-100k/ua.test', sep='\t', header=None,
                             names=['user_id', 'item_id', 'rating', 'timestamp'])
    user_data = pd.read_csv('DGL/ml-100k/u.user', sep='|', header=None, encoding='latin1')
    item_data = pd.read_csv('DGL/ml-100k/u.item', sep='|', header=None, encoding='latin1')

    # 测试集和训练集的用户项目不同，根据训练集对测试集进行精简
    test_data = test_data[test_data['user_id'].isin(train_data['user_id']) &
                          test_data['item_id'].isin(train_data['item_id'])]

    return train_data, test_data, user_data, item_data

def get_fshl():
    dbHandle = DatabaseIo()
    if not dbHandle:
        return None

    # sql_user = """SELECT id,sex FROM account5000"""
    sql_user = DataBaseQuery["dgl_user_info"]
    # sql_course = "select id , system_course_id ,course_name from course_info"
    # sql_course = """select id,classify_id from course5000"""
    sql_course = DataBaseQuery["feature_classify"]
    # sql_user = """select user_id from user_basic_info"""
    # sql_dr = """select * from course_dr5000"""
    sql_dr = DataBaseQuery["course_dr"]

    result_dr = dbHandle.doSql(execType=DataBaseOperateType.SearchMany,
                               sql=sql_dr)
    result_course = dbHandle.doSql(execType=DataBaseOperateType.SearchMany,
                                   sql=sql_course)
    result_user = dbHandle.doSql(execType=DataBaseOperateType.SearchMany,
                                 sql=sql_user)
    print(result_dr)

    # 数据加载
    train_data = pd.DataFrame(list(result_dr))
    train_data.columns = ['user_id', 'item_id', 'rating']
    user_data = pd.DataFrame(list(result_user))
    item_data = pd.DataFrame(list(result_course))

    return train_data,user_data,item_data

def get_bigraph():
    dbHandle = DatabaseIo()
    if not dbHandle:
        return None

    sql_user = DataBaseQuery["dgl_user_info"]

    sql_course = DataBaseQuery["feature_classify"]

    sql_classify = DataBaseQuery["classify_info"]

    result_course = dbHandle.doSql(execType=DataBaseOperateType.SearchMany,
                                   sql=sql_course)
    result_user = dbHandle.doSql(execType=DataBaseOperateType.SearchMany,
                                 sql=sql_user)
    result_classify = dbHandle.doSql(execType=DataBaseOperateType.SearchMany,
                                     sql=sql_classify)
    # 数据加载
    train_data = pd.read_csv('file_saved/toGcn.csv', header=None,
                             names=['user_id', 'item_id', 'rating'])
    train_data.columns = ['user_id', 'item_id', 'rating']
    user_data = pd.DataFrame(list(result_user))
    item_data = pd.DataFrame(list(result_course))
    classify_data = pd.DataFrame(list(result_classify))
    classify_data.columns = ['id', 'course_name', 'classify_name', 'classify_id']


    return train_data, user_data, item_data, classify_data



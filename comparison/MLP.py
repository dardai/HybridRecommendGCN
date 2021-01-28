#!/usr/bin/env python 
# -*- coding:utf-8 -*-

from tqdm import tqdm_notebook
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Multiply, Concatenate
# from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# import tensorflow.keras.backend as K
import pandas as pd
# import numpy as np
from globalConst import DataBaseQuery, DataBaseOperateType, SetType
from utils.databaseIo import DatabaseIo
from utils.extends import formatDataByType
import random


def loadData(flag = True):
    if flag == True:
        all_data = pd.read_csv('../dgl/ml-100k/u.data', sep='\t', header=None,
                               names=['user_id', 'item_id', 'rating', 'timestamp'])
        # test_data = pd.read_csv('../dgl/ml-100k/ua.test', sep='\t', header=None,
        #                         names=['user_id', 'item_id', 'rating', 'timestamp'])
        user_data = pd.read_csv('../dgl/ml-100k/u.user', sep='|', header=None, encoding='latin1')
        item_data = pd.read_csv('../dgl/ml-100k/u.item', sep='|', header=None, encoding='latin1')
        # test_data = test_data[test_data['user_id'].isin(train_data['user_id']) &
        #                       test_data['item_id'].isin(train_data['item_id'])]
        # u_data = user_data[[0,1,2,3,4]]
        # u_data.columns = ['user_id','age','gender','occupation','zip_code']
        # i_data = item_data
        # i_data.columns = ['item_id','title','release_date','video_release_date','IMDb_URL','unknown','Action','Adventure','Animation','Children',
        #                   'Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller',
        #                   'War','Western']
    else:
        dbHandle = DatabaseIo()
        if not dbHandle:
            return None
        sql_dr = DataBaseQuery["course_dr"]
        sql_course = DataBaseQuery["course_info"]
        sql_user = DataBaseQuery["user_id"]
        result_dr = dbHandle.doSql(execType=DataBaseOperateType.SearchMany,
                                   sql=sql_dr)
        result_course = dbHandle.doSql(execType=DataBaseOperateType.SearchMany,
                                       sql=sql_course)
        dbHandle.changeCloseFlag()
        result_user = dbHandle.doSql(execType=DataBaseOperateType.SearchMany,
                                     sql=sql_user)
        drList = formatDataByType(SetType.SetType_List, result_dr)
        all_data = drList
        user_data = formatDataByType(SetType.SetType_Set, result_user)
        item_data = formatDataByType(SetType.SetType_List, result_course)

    # return all_data,user_data,item_data,u_data,i_data
    return all_data,user_data,item_data

def MLP(all_data, u_data, i_data,epoch,batch_size):
    # all_data, user_data, item_data, u_data, i_data = loadData()
    # df_items = item_data[['item_id', 'title']]
    # print('Movies:')
    # print(df_items.head())

    # print('Ratings')
    df_ratings = all_data
    # print(df_ratings.head())
    df_ratings = df_ratings[['user_id', 'item_id', 'rating']]

    num_users = len(df_ratings.user_id.unique())
    num_items = len(df_ratings.item_id.unique())
    print('There are {} unique users and {} unique movies in this data set'.format(num_users, num_items))

    # print(df_ratings.head())

    num_users = len(df_ratings.user_id.unique())
    num_items = len(df_ratings.item_id.unique())
    print('There are {} unique users and {} unique movies in this data set'.format(num_users, num_items))

    user_maxId = df_ratings.user_id.max()
    item_maxId = df_ratings.item_id.max()
    print('There are {} distinct users and the max of user ID is also {}'.format(num_users, user_maxId))
    print('There are {} distinct movies, however, the max of movie ID is {}'.format(num_items, item_maxId))
    print('In the context of matrix factorization, the current item vector is in unnecessarily high dimensional space')
    print('So we need to do some data cleaning to reduce the dimension of item vector back to {}'.format(num_items))

    users = {}
    k = 0
    for user in tqdm_notebook(df_ratings.user_id.unique()):
        users[user] = k
        k += 1

    movies = {}
    k = 0
    for movie in tqdm_notebook(df_ratings.item_id.unique()):
        movies[movie] = k
        k += 1

    df_ratings_f = df_ratings.copy()
    df_ratings_f['user_id'] = df_ratings['user_id'].map(users)
    df_ratings_f['item_id'] = df_ratings['item_id'].map(movies)

    df_ratings_f.head().reset_index(drop=True)

    train, test = train_test_split(df_ratings_f, test_size=0.2, shuffle=False, random_state=99)

    user = Input(shape=(1,))
    item = Input(shape=(1,))

    embed_user = Embedding(input_dim=num_users + 1, output_dim=32, embeddings_initializer='uniform',
                           name='user_embedding', input_length=1)(user)
    embed_item = Embedding(input_dim=num_items + 1, output_dim=32, embeddings_initializer='uniform',
                           name='item_embedding', input_length=1)(item)

    user2 = Flatten()(embed_user)
    item2 = Flatten()(embed_item)

    combine = Concatenate(axis=-1)([user2, item2])

    layer1 = Dense(32, activation='relu', kernel_initializer='glorot_uniform')(combine)
    layer2 = Dense(32, activation='relu', kernel_initializer='glorot_uniform')(layer1)
    layer3 = Dense(32, activation='relu', kernel_initializer='glorot_uniform')(layer2)

    out = Dense(1)(layer3)

    model = Model([user, item], out)
    model.compile(loss="mean_squared_error", optimizer="adam")
    # model.summary()

    model.fit([train.user_id.values, train.item_id.values], train.rating.values, epochs=epoch, batch_size=batch_size, verbose=2)

    # pred = model.predict([test.user_id.values, test.item_id.values])
    u_pre = u_data[0].values.tolist()
    i_pre = i_data[0].values.tolist()
    ilen = len(i_pre)
    ulen = len(u_pre)
    u_pre = u_pre * ilen
    i_pre = i_pre * ulen
    pre = pd.DataFrame({'user_id': u_pre, 'item_id': i_pre})
    pred = model.predict([pre.user_id.values, pre.item_id.values])

    pre['rating'] = pred
    result = pre.groupby('user_id').apply(lambda x: x.sort_values(by="rating", ascending=False)).reset_index(drop=True)

    print(result)
    result.to_csv('../file_saved/MLPresult.csv',index=None)
    return result

def makeClassifyDict(item_data):
    print("make classify dict")
    # 示例{“课程id”：[类别1、类别2、类别3]}
    classifydict = {}
    item = item_data
    item = item.values.tolist()
    for row in item:
        if row[0] not in classifydict.keys():
            classifydict[row[0]] = []
            for i in range(5, 24):
                if row[i] == 1:
                    classifydict[row[0]].append(i)
        else:
            continue
    return classifydict

def recommend(item_data,topK):
    result = pd.read_csv('../file_saved/MLPresult.csv')
    result_topK = result.groupby(['user_id']).head(topK).reset_index(drop=True)

    # 计算列表覆盖率
    recommend_length = len(result_topK['item_id'].value_counts())
    item_length = len(item_data)
    print(recommend_length/item_length)

    # 计算物品类别覆盖率
    # 记录推荐类别的字典
    classify_num_dict = {}
    classify = makeClassifyDict(item_data)
    item_id = result_topK['item_id'].value_counts().values.tolist()
    for row in item_id:
        for c in classify[row]:
            if isinstance(c, int):
                if c not in classify_num_dict.keys():
                    classify_num_dict[c] = 1
                else:
                    continue
            else:
                for i in c:
                    if i not in classify_num_dict.keys():
                        classify_num_dict[i] = 1
                    else:
                        continue
    classify_cov = (len(classify_num_dict) * 1.0) / 19.0
    print(classify_cov)



all_data, user_data, item_data = loadData()
MLP(all_data, user_data, item_data,epoch=10,batch_size=32)
recommend(item_data,topK=10)
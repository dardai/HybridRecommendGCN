# -*- coding: utf-8 -*-
import math
import random

import pandas as pd

from globalConst import DataBaseQuery, DataBaseOperateType, SetType
from utils.databaseIo import DatabaseIo
from utils.extends import formatDataByType


def getdata(flag = True):
    print("get data")
    if flag == True:
        train_data = pd.read_csv('../dgl/ml-100k/ua.base', sep='\t', header=None,
                                 names=['user_id', 'item_id', 'rating', 'timestamp'])
        test_data = pd.read_csv('../dgl/ml-100k/ua.test', sep='\t', header=None,
                                names=['user_id', 'item_id', 'rating', 'timestamp'])
        user_data = pd.read_csv('../dgl/ml-100k/u.user', sep='|', header=None, encoding='latin1')
        item_data = pd.read_csv('../dgl/ml-100k/u.item', sep='|', header=None, encoding='latin1')

        train_data = train_data.values.tolist()
        test_data = test_data.values.tolist()
        user_data = user_data.values.tolist()
        item_data = item_data.values.tolist()
        # print item_data
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
        user_data = formatDataByType(SetType.SetType_Set, result_user)
        item_data = formatDataByType(SetType.SetType_List, result_course)
        dr_length = len(drList)
        testIDs = random.sample(range(1, dr_length), int(dr_length / 10))
        data = pd.DataFrame(drList)
        train_data = []
        test_data = []
        for index, row in data.iterrows():
            if (index + 1) in testIDs:
                test_data.append(row)
            else:
                train_data.append(row)

    return train_data, test_data, user_data, item_data

def createDict(data):
    print("make user_dict and item_dict")
    user_dict = {}
    item_dict = {}
    for recode in data:
        if recode[0] in user_dict:
            user_dict[recode[0]].append((recode[1], recode[2]))
        else:
            user_dict[recode[0]] = [(recode[1], recode[2])]
        if recode[1] in item_dict:
            item_dict[recode[2]].append(recode[0])
        else:
            item_dict[recode[1]] = [recode[0]]
    return user_dict, item_dict

def ItemSimilarity(user_dict):
    # 记录每个物品被多少个用户交互的数量
    N = {}
    # 记录两个物品的共现次数
    C = {}
    # 记录物品与物品之间的相似度，即物品-物品的相似度矩阵
    W = {}
    # 记录W中每一列的最大值， 按列进行归一化
    W_max = {}
    print("make dict N and dict C")
    for key in user_dict:
        for i in user_dict[key]:
            # i[0]为物品的id
            # 初始化字典N
            if i[0] not in N.keys():
                N[i[0]] = 0
            N[i[0]] = N[i[0]] + 1
            C[i[0]] = {}

            for j in user_dict[key]:
                if i == j:
                    continue
                # j[0]为物品的id
                # 初始化字典C[i[0]]
                if j[0] not in C[i[0]].keys():
                    C[i[0]][j[0]] = 0
                C[i[0]][j[0]] = C[i[0]][j[0]] + 1 / math.log(1 + len(user_dict[key]) * 1.0)

    print("make dict W")
    for i, related_items in C.items():
        W[i] = {}
        for j, cij in related_items.items():
            if j not in W_max.keys():
                W_max[j] = 0.0
            W[i][j] = cij / (math.sqrt(N[i] * N[j]))
            if W[i][j] > W_max[j]:
                W_max[j] = W[i][j]

    for i, related_items in C.items():
        for j, cij in related_items.items():
            W[i][j] = W[i][j] / W_max[j]

    return W

def makeRecommend(user_dict, K):
    print("make recommend")
    rank = {}
    result = []
    W = ItemSimilarity(user_dict)
    for user_id in user_dict.keys():
        for i, score in user_dict[user_id]:
            for j, wj in sorted(W[i].items(), key = lambda x:x[1], reverse = True)[0 : K]:
                if j in user_dict[user_id]:
                    continue
                if j not in rank.keys():
                    rank[j] = 0
                rank[j] = rank[j] + score * wj
        temp_dict = dict(sorted(rank.items(), key = lambda x:x[1], reverse = True)[0 : K])

        for key , value in temp_dict.items():
            temp_list = []
            temp_list.append(user_id)
            temp_list.append(key)
            temp_list.append(value)
            result.append(temp_list)
    return result

train_data, test_data, user_data, item_data = getdata(False)
user_dict, item_dict = createDict(train_data)
result = makeRecommend(user_dict, 10)
print result
recommend_result = pd.DataFrame(result)
recommend_result.to_csv("../file_saved/itemCF.csv", header = False, index = False)



# getdata()
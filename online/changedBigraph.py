#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import random
import codecs
import logging

from decimal import Decimal
from utils.extends import formatDataByType, makeDic
from globalConst import DataBaseOperateType, SetType
from changedPredeal import updateCourseDrChanged
from utils.databaseIo import DatabaseIo

def getDataFromDB():
    dbHandle = DatabaseIo()
    if not dbHandle:
        return None

    sql_dr = """SELECT * FROM course_dr5000_changed"""
    # sql_course = "select id , system_course_id ,course_name from course_info"
    sql_course = """select id, name from course5000"""
    # sql_user = """select user_id from user_basic_info"""
    sql_user = """select id from account5000"""

    result_dr = dbHandle.doSql(execType=DataBaseOperateType.SearchMany,
                               sql=sql_dr)
    result_course = dbHandle.doSql(execType=DataBaseOperateType.SearchMany,
                                   sql=sql_course)
    dbHandle.changeCloseFlag()
    result_user = dbHandle.doSql(execType=DataBaseOperateType.SearchMany,
                                 sql=sql_user)

    print("len(result_dr) = {}, len(result_user) = {},\
          len(result_course) = {}".format(len(result_dr),
                                          len(result_user),
                                          len(result_course)))

    return result_dr, result_course, result_user
# 读取推荐课程的名字
def get_keys(value, courseList):
    for row in courseList:
        if row[0] == value:
            return row[2]

def dataPreprocessiong():
    result_dr, result_course, result_user = getDataFromDB()

    drList = formatDataByType(SetType.SetType_List, result_dr)
    userList = formatDataByType(SetType.SetType_Set, result_user)
    courseList = formatDataByType(SetType.SetType_List, result_course)


    dr_length = len(drList)
    course_length = len(courseList)
    user_length = len(userList)

    user_mdic, user_mdicr = makeDic(userList)
    course_mdic, course_mdicr = makeDic(courseList)

    result, learned = [], []
    for dr in drList:
        temp_result, temp_learned = [], []
        temp_result.append(user_mdic[dr[0]] + 1)
        temp_result.append(dr[1])
        temp_result.append(dr[2] * 5)
        temp_learned.append(dr[0])
        temp_learned.append(dr[1])
        temp_learned.append(get_keys(dr[1], courseList))
        result.append(temp_result)
        learned.append(temp_learned)

    data = pd.DataFrame(result)


    return data, learned, course_mdic, course_mdicr, user_mdic, user_mdicr, \
           dr_length, course_length, user_length, courseList


def makeTrainMatrix(data, course_length, user_length, dr_length, course_mdic):
    all_rated_graph = np.zeros([course_length, user_length])  # 创建所有已评价矩阵
    train_graph = np.zeros([course_length, user_length])  # 创建训练图矩阵
    test_graph = np.zeros([course_length, user_length])  # 创建测试图矩阵
    train_rated_graph = np.zeros([course_length, user_length])  # 创建训练集里已评价矩阵
    testIDs = random.sample(range(1, dr_length), int(dr_length / 10))

    for index, row in data.iterrows():
        if ((index + 1) in testIDs):
            test_graph[course_mdic[row[1]], int(row[0]) - 1] = 1
            all_rated_graph[course_mdic[row[1]], int(row[0]) - 1] = 1
        else:
            train_rated_graph[course_mdic[row[1]], int(row[0]) - 1] = 1
            all_rated_graph[course_mdic[row[1]], int(row[0]) - 1] = 1

            if (int(row[2]) >= 3.0):
                train_rated_graph[course_mdic[row[1]], int(row[0]) - 1] = row[2]
                train_graph[course_mdic[row[1]], int(row[0]) - 1] = 1

    return all_rated_graph, train_graph, test_graph, train_rated_graph


def doBigraph():
    data, learned, course_mdic, course_mdicr, \
    user_mdic, user_mdicr, dr_length, course_length, \
    user_length, courseList = dataPreprocessiong()

    all_rated_graph, train_graph, test_graph, train_rated_graph = \
        makeTrainMatrix(data, course_length, user_length,
                        dr_length, course_mdic)

    # 为资源配置矩阵做准备
    kjs = np.zeros([course_length])
    kls = np.zeros([user_length])

    # 求产品的度
    for rid in range(course_length):
        kjs[rid] = train_graph[rid, :].sum()

    # 求用户的度
    for cid in range(user_length):
        kls[cid] = train_graph[:, cid].sum()

    # 计算每个用户未选择产品的度
    s = np.ones(user_length)
    s *= course_length
    ls = s - kls

    # 为防止之后的除法出现0，手动将其改为极大值
    for i in range(course_length):
        if (kjs[i] == 0.0):
            kjs[i] = 99999
    for i in range(user_length):
        if (kls[i] == 0.0):
            kls[i] = 99999

    # 求资源配额矩阵
    weights = np.zeros([course_length, course_length])
    # 转换为矩阵乘法和向量除法
    # 设定若干中间值
    gt = train_graph.T
    temp = np.zeros([user_length, course_length])
    for i in range(course_length):
        temp[:, i] = gt[:, i] / kls

    # temp = nm.array(sparkMultiply(train_graph, temp,
    #                              user_length, course_length))
    temp = np.matmul(train_graph, temp)
    for i in range(course_length):
        weights[i, :] = temp[i, :] / kjs

    # 求各个用户的资源分配矩阵
    locate = np.matmul(weights, train_rated_graph)
    # locate = nm.array(sparkMultiply(weights, train_rated_graph,
    #                                course_length, user_length))
    # 将算法产生的推荐结果以列表形式存储
    recommend = []
    for i in range(len(locate)):
        for j in range(len(locate[i])):
            # 不能过滤用户已学习过的课程，注释掉这个条件
            # if all_rated_graph[i][j] != 1:
            data = [j + 1, i + 1, locate[i][j]]
            temp_data = list(data)
            recommend.append(temp_data)

    recommend_result = []
    for i5 in recommend:
        if i5[2] > 0.0:
            po = []
            po.append(user_mdicr[i5[0] - 1])
            po.append(course_mdicr[i5[1] - 1])
            po.append(get_keys(course_mdicr[i5[1] - 1], courseList))
            # 格式化推荐度的值
            po.append(Decimal(i5[2]).quantize(Decimal('0.00000')))
            recommend_result.append(tuple(po))

    return locate, recommend_result, learned, user_length, ls, test_graph

def storeData(recommend_result):
    #修改文件目录
    #myfile = codecs.open("C:/Users/Administrator/Desktop/HybridRecommendGCN/online/changedBigraph.csv", mode="w", encoding='utf-8')
    myfile = codecs.open("changedBigraph.csv", mode="w", encoding='utf-8')
    result_data = sorted(tuple(recommend_result))
    df = pd.DataFrame(result_data)

    #在设置的时间内没有新增交互数据
    if df.empty:
        print "no changed data"
        return

    df = df.drop(2,axis=1)
    df.sort_values([0,3],ascending = [1,0],inplace=True)
    grouped = df.values.tolist()
    #按四舍五入处理推荐值，不保存推荐值为0的数据
    for row in grouped:
        tempRow = row[2]
        if (tempRow > 5):
            tempRow = 5
        tempRow = round(tempRow)
        tempRow = int(tempRow)
        if tempRow != 0 :
            myfile.write(str(row[0]))
            myfile.write(",")
            myfile.write(str(row[1]))
            myfile.write(",")
            myfile.write(str(tempRow))
            myfile.write("\n")
    myfile.close()

def bigraphChangedMain():
    logging.warning("运行日志：在线模块二部图")
    print("run bigraph(changed)...")
    locate, recommend_result, learned, \
    user_length, ls, test_graph = doBigraph()
    storeData(recommend_result)
    print("bigraph(changed) success")

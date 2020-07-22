#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import numpy as nm
import random


def saveBgInputMartix(data,user_mdicr):
    # 构造二部图的输入数据的真实值矩阵realGraph,以及长id到短id(0开始)的字典
    userListNew = list(data[0])
    for i in range(len(userListNew)):
        userListNew[i] = user_mdicr[userListNew[i] - 1]
    userListNewTemp = sorted(list(set(userListNew)))
    courseListNew = list(data[1])
    courseListNewTemp = sorted(list(set(courseListNew)))
    course_mdic_new = {}
    user_mdic_new = {}

    for i in range(len(userListNewTemp)):
        user_mdic_new[userListNewTemp[i]] = i
    for i in range(len(courseListNewTemp)):
        course_mdic_new[courseListNewTemp[i]] = i

    realGraph = nm.zeros((len(courseListNewTemp), len(userListNewTemp)))
    for i in range(len(data)):
        realGraph[[course_mdic_new[courseListNew[i]]], user_mdic_new[userListNew[i]]] = 1

    nm.savetxt('C:/Users/Administrator/Desktop/HybridRecommendGCN/roc/realGraph.txt', realGraph)
    nm.save('user_mdic_new.npy', user_mdic_new)
    nm.save('course_mdic_new.npy', course_mdic_new)

#切分的数据，用于画ROC的计算
def makeTrainMatrixSplit(data, course_length, user_length, dr_length, course_mdic):
    all_rated_graph = nm.zeros([course_length, user_length])    # 创建所有已评价矩阵
    train_graph = nm.zeros([course_length, user_length])    # 创建训练图矩阵
    test_graph = nm.zeros([course_length, user_length])    # 创建测试图矩阵
    train_rated_graph = nm.zeros([course_length, user_length])    # 创建训练集里已评价矩阵
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

def rocLocate(data,course_length,user_length,dr_length,course_mdic):
    all_rated_graph, train_graph, test_graph, train_rated_graph = \
        makeTrainMatrixSplit(data, course_length, user_length,
                        dr_length, course_mdic)
    # 为资源配置矩阵做准备
    kjs = nm.zeros([course_length])
    kls = nm.zeros([user_length])

    # 求产品的度
    for rid in range(course_length):
        kjs[rid] = train_graph[rid, :].sum()

    # 求用户的度
    for cid in range(user_length):
        kls[cid] = train_graph[:, cid].sum()

    # 计算每个用户未选择产品的度
    s = nm.ones(user_length)
    s *= course_length
    # ls = s - kls

    # 为防止之后的除法出现0，手动将其改为极大值
    for i in range(course_length):
        if (kjs[i] == 0.0):
            kjs[i] = 99999
    for i in range(user_length):
        if (kls[i] == 0.0):
            kls[i] = 99999

    # 求资源配额矩阵
    weights = nm.zeros([course_length, course_length])
    # 转换为矩阵乘法和向量除法
    # 设定若干中间值
    gt = train_graph.T
    temp = nm.zeros([user_length, course_length])
    for i in range(course_length):
        temp[:, i] = gt[:, i] / kls

    # temp = nm.array(sparkMultiply(train_graph, temp,
    #                              user_length, course_length))
    temp = nm.matmul(train_graph, temp)
    for i in range(course_length):
        weights[i, :] = temp[i, :] / kjs

    # 求各个用户的资源分配矩阵
    locate = nm.matmul(weights, train_rated_graph)

    nm.savetxt("C:/Users/Administrator/Desktop/HybridRecommendGCN/roc/bgLocate.txt", locate, delimiter=',')
    nm.savetxt("C:/Users/Administrator/Desktop/HybridRecommendGCN/roc/test_graph.txt", test_graph, delimiter=',')



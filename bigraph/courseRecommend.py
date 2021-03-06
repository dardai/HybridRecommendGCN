﻿# -*- coding: utf-8 -*-

# 周洋涛-2019.9
# 本代码实现了二部图算法，并用批量存储的方法将产生的推荐结果存入到SQL server数据库中的course_model表中
import sys
import os
project = '\\Desktop\\HybridRecommendGCN'  # 工作项目根目录
sys.path.append(os.getcwd().split(project)[0] + project)
import prettytable as pt
from decimal import Decimal
# from sklearn.metrics import roc_curve, auc
import pandas as pd
import numpy as nm
import random
import codecs
from globalConst import DataBaseOperateType, SetType, DataBaseQuery
from utils.extends import formatDataByType, makeDic
from utils.databaseIo import DatabaseIo
from roc.rocUtils import saveBgInputMartix,rocLocate
from bgUtils import storeDataAsGCNInput
import logging
import scipy.sparse as sp

# 定义将多条数据存入数据库操作
drawRoc = False
getGcnInput = True
# 输入：由pysaprk中的行矩阵rdd转换成的列表，形如
# [ DenseMatrix[1,1,1], DenseMatrix[1,1,1], DenseMatrix[1,1,1] ]
# 返回：转换成功的列表，形如[ [1,1,1], [1,1,1], [1,1,1] ]
def transToMatrix(p):
    return formatDataByType(SetType.SetType_Set, p)


# 输入：要进行矩阵乘法运算的c和d数组，rlength是数组矩阵d的行数，clength是数组矩阵d的列数
# 返回：实现矩阵乘法的结果矩阵
# 注意：返回的结果矩阵是numpy.matrix类型，但二部图中定义的矩阵都是numpy.array类型，要对函数返回的结果进行类型转换
'''
def sparkMultiply(c, d, rlength, clength):
    # 将d数组里的所有行数组合并成一个大数组
    b2 = _flatten(d.tolist())
    # 设置spark相关参数
    sc = SparkContext('local', 'tests')
    # 进行并行化
    t1 = sc.parallelize(c.tolist())
    # t2 = sc.parallelize(d)
    # 创建行矩阵
    m1 = RowMatrix(t1)
    # 创建密集矩阵，由于pyspark中的矩阵都是按列存储，所以这里参数设置为True使得矩阵创建时与numpy一样按行存储
    m2 = DenseMatrix(rlength, clength, list(b2), True)
    # 调用pyspark中的矩阵乘法，注意这里的m2一定要对应输入时的d数据矩阵
    mat = m1.multiply(m2)
    # print(mat.rows.collect())
    # 下面两行代码实现将RDD类型转换成列表类型
    k = mat.rows.collect()
    q = transToMatrix(k)
    # 结束并行化
    sc.stop()
    print(q)
    return q
'''


def getDataFromDB():
    logging.warning(u"运行日志：从数据库中读取交互数据、用户数据、课程数据")
    dbHandle = DatabaseIo()
    if not dbHandle:
        return None

    # sql_dr = """SELECT * FROM course_dr5000"""
    sql_dr = DataBaseQuery["course_dr"]
    #sql_course = "select id , system_course_id ,course_name from course_info"
    # sql_course = """select id, name from course5000"""
    sql_course = DataBaseQuery["course_info"]
    #sql_user = """select user_id from user_basic_info"""
    # sql_user = """select id from account5000"""
    sql_user = DataBaseQuery["user_id"]

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
    # logging.warning(u"运行日志：根据索引读取课程的名字")
    for row in courseList:
        if row[0] == value:
            #return row[2]
            return row[1]


def dataPreprocessiong():
    logging.warning(u"运行日志：将从数据库中读出的数据进行数据处理")
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
        #print dr[0]
        temp_result.append(user_mdic[dr[0]] + 1)
        temp_result.append(dr[1])
        temp_result.append(dr[2] * 5)
        temp_learned.append(dr[0])
        temp_learned.append(dr[1])
        temp_learned.append(get_keys(dr[1], courseList))
        result.append(temp_result)
        learned.append(temp_learned)

    data = pd.DataFrame(result)

    if drawRoc:
        saveBgInputMartix(data,user_mdicr)

    return data, learned, course_mdic, course_mdicr, user_mdic, user_mdicr, \
           dr_length, course_length, user_length, courseList


def makeTrainMatrix(data, course_length, user_length, dr_length, course_mdic):
    logging.warning(u"运行日志：构建训练和测试矩阵")
    # all_rated_graph = nm.zeros([course_length, user_length])  # 创建所有已评价矩阵
    # train_graph = nm.zeros([course_length, user_length])  # 创建训练图矩阵
    # test_graph = nm.zeros([course_length, user_length])  # 创建测试图矩阵
    # train_rated_graph = nm.zeros([course_length, user_length])  # 创建训练集里已评价矩阵
    all_rated_graph = sp.lil_matrix((course_length, user_length))  # 创建所有已评价压缩矩阵
    train_graph = sp.lil_matrix((course_length, user_length))  # 创建训练图压缩矩阵
    test_graph = sp.lil_matrix((course_length, user_length))  # 创建测试图压缩矩阵
    train_rated_graph = sp.lil_matrix((course_length, user_length))  # 创建训练集里已评价压缩矩阵
    testIDs = random.sample(range(1, dr_length), int(dr_length / 10))
    #如果需要得到可以输入gcn的数据，就不切分数据集
    if getGcnInput:
        for index, row in data.iterrows():
            test_graph[course_mdic[row[1]], int(row[0]) - 1] = 1
            train_rated_graph[course_mdic[row[1]], int(row[0]) - 1] = 1
            all_rated_graph[course_mdic[row[1]], int(row[0]) - 1] = 1

            if (int(row[2]) >= 3.0):
                train_rated_graph[course_mdic[row[1]], int(row[0]) - 1] = 1
                # train_rated_graph[course_mdic[row[1]], int(row[0]) - 1] = row[2]
                train_graph[course_mdic[row[1]], int(row[0]) - 1] = 1
    else:
        for index, row in data.iterrows():
            if ((index + 1) in testIDs):
                test_graph[course_mdic[row[1]], int(row[0]) - 1] = 1
                all_rated_graph[course_mdic[row[1]], int(row[0]) - 1] = 1
                #train_rated_graph[course_mdic[row[1]], int(row[0]) - 1] = 1
                #if (int(row[2]) >= 3.0):
                #    train_rated_graph[course_mdic[row[1]], int(row[0]) - 1] = 1.5
            else:
                train_rated_graph[course_mdic[row[1]], int(row[0]) - 1] = 1
                all_rated_graph[course_mdic[row[1]], int(row[0]) - 1] = 1

                if (int(row[2]) >= 3.0):
                    #train_rated_graph[course_mdic[row[1]], int(row[0]) - 1] = 1.5
                    train_rated_graph[course_mdic[row[1]], int(row[0]) - 1] = 1
                    train_graph[course_mdic[row[1]], int(row[0]) - 1] = 1

    # 转换成csr_matrix
    all_rated_graph = all_rated_graph.tocsr()
    train_graph = train_graph.tocsr()
    test_graph = test_graph.tocsr()
    train_rated_graph = train_rated_graph.tocsr()

    return all_rated_graph, train_graph, test_graph, train_rated_graph


def doBigraph():
    logging.warning(u"运行日志：二部图模型构建")
    data, learned, course_mdic, course_mdicr, \
    user_mdic, user_mdicr, dr_length, course_length, \
    user_length, courseList = dataPreprocessiong()

    all_rated_graph, train_graph, test_graph, train_rated_graph = \
        makeTrainMatrix(data, course_length, user_length,
                        dr_length, course_mdic)
    if drawRoc:
        rocLocate(data,course_length,user_length,dr_length,course_mdic)
    # 为资源配置矩阵做准备
    kjs = nm.zeros([course_length])
    kls = nm.zeros([user_length])

    # 求产品的度
    for rid in range(course_length):
        kjs[rid] = train_graph.getrow(rid).sum()
        # kjs[rid] = train_graph[rid, :].sum()

    # 求用户的度
    for cid in range(user_length):
        kls[cid] = train_graph.getcol(cid).sum()
        # kls[cid] = train_graph[:, cid].sum()

    # 计算每个用户未选择产品的度
    s = nm.ones(user_length)
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
    # weights = sp.lil_matrix((course_length, course_length))
    # weights = nm.zeros([course_length, course_length])
    # 转换为矩阵乘法和向量除法
    # 设定若干中间值
    gt = train_graph.transpose()
    gi = gt.getcol(0)
    kls = kls.reshape(user_length, 1)
    temp = gi / kls
    temp = temp.reshape(user_length, 1)
    temp = sp.csr_matrix(temp)
    # temp = nm.zeros([user_length, course_length])
    for i in range(1, course_length):
        gi = gt.getcol(i)
        t = gi / kls
        t = t.reshape(user_length, 1)
        t = sp.csr_matrix(t)
        temp = sp.hstack([temp, t])
        # temp[:, i] = gt[:, i] / kls
    temp = sp.csr_matrix(temp)

    # temp = nm.array(sparkMultiply(train_graph, temp,
    #                              user_length, course_length))
    # temp = nm.matmul(train_graph, temp)
    temp = train_graph.dot(temp)
    temp = sp.csr_matrix(temp)
    tempi = temp.getrow(0)
    kjs = kjs.reshape(1, course_length)
    weights = tempi / kjs
    weights = weights.reshape(1, course_length)
    weights = sp.csr_matrix(weights)
    for i in range(1, course_length):
        tempi = temp.getrow(i)
        w = tempi / kjs
        w = w.reshape(1, course_length)
        w = sp.csr_matrix(w)
        weights = sp.vstack([weights, w])
        # weights[i, :] = temp[i, :] / kjs
    weights = sp.csr_matrix(weights)

    # 求各个用户的资源分配矩阵
    # locate = nm.matmul(weights, train_rated_graph)
    locate = weights.dot(train_rated_graph)
    locate = sp.csr_matrix(locate).toarray()
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
    logging.warning(u"运行日志：二部图推荐结果保存")
    # myfile = codecs.open("data.txt", mode="w", encoding='utf-8')
    myfile = codecs.open("file_saved/data.txt", mode="w", encoding='utf-8')
    result_data = sorted(tuple(recommend_result))
    myfile.write("user_id")
    myfile.write("   ")
    myfile.write("course_id")
    myfile.write("   ")
    myfile.write("course_name")
    myfile.write("   ")
    myfile.write("recommend_value")
    myfile.write("\n")

    for row in result_data:
        myfile.write(str(row[0]))
        myfile.write(" ")
        myfile.write(str(row[1]))
        myfile.write(" ")
        myfile.write(row[2])
        myfile.write(" ")
        myfile.write(str(row[3]))
        myfile.write("\n")

    myfile.close()


def bigraphMain():
    logging.warning(u"运行日志：进行二部图运算")
    print("run bigraph...")
    locate, recommend_result, learned, \
    user_length, ls, test_graph = doBigraph()
    storeData(recommend_result)
    #转换为GCN可输入的数据
    if getGcnInput:
        storeDataAsGCNInput(recommend_result)

    result_data = sorted(recommend_result, key=lambda x: x[0] and x[1])
    result_data = sorted(result_data, key=lambda x: x[3], reverse=True)
    # 将产生的推荐结果以id, course_index, recommend_value存入数据库中的course_model表中
    # insertData(tuple2)
    # 开始求预测的准确性
    # rs = nm.zeros(user_length)  # ?? 有什么用

    # 求测试集中电影的排名矩阵

    # 得到资源配置矩阵对应的排名
    indiceLocate = nm.argsort(locate, axis=0)
    indiceLocate = sp.csr_matrix(indiceLocate)

    # 通过矩阵对应元素相乘得到测试集的排名数据
    # 为方便后续处理，对结果进行转置
    # testIndice = (indiceLocate * test_graph).T
    testIndice = indiceLocate.multiply(test_graph)
    testIndice = sp.csr_matrix(testIndice).transpose().toarray()
    # 求精确度的值
    usum = 0
    # 计算测试集中每部已评分电影的距离，并求均值
    for i in range(user_length):
        if (test_graph[:, i].sum() > 0):
            usum += ((testIndice[i]).sum() / (ls[i] * test_graph[:, i].sum()))
    #print("the average value of r is:")
    #print(usum / user_length)
    print("bigraph success")

    return learned, result_data, test_graph


def Main():
    logging.warning(u"运行日志：对二部图推荐结果进行循环id查询展示")
    learned, result_data, test_graph = bigraphMain()
    while True:
        user_id = input("请输入用户id：")
        if user_id == "quit":
            break
        else:
            ta = pt.PrettyTable()
            ta.field_names = ["User_id", "Course_id", "Course_name"]
            for row in learned:
                if row[0] == int(user_id):
                    ta.add_row(row)
            print("该用户已学习过的课程有：")
            print(ta)
            tb = pt.PrettyTable()
            tb.field_names = ["User_id", "Course_id", "Course_name",
                              "Recommend_value"]
            for row in result_data:
                if row[0] == int(user_id):
                    tb.add_row(row)
            print("为该用户推荐学习的课程有：")
            print(tb)


# bigraphMain()
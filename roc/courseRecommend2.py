# -*- coding: utf-8 -*-

# 周洋涛-2019.9
# 本代码实现了二部图算法，并用批量存储的方法将产生的推荐结果存入到SQL server数据库中的course_model表中
# import databaseIo
import prettytable as pt
from decimal import Decimal
# from sklearn.metrics import roc_curve, auc
import pandas as pd
import numpy as nm
import random
import codecs
import math
# import io
# 引入pyspark的相关包
# from pyspark import SparkContext
# from pyspark.mllib.linalg import Matrix
# from pyspark.mllib.linalg.distributed import RowMatrix, DenseMatrix
# from tkinter import _flatten
# from Tkinter import _flatten

from globalConst import DataBaseOperateType, SetType
from utils.extends import formatDataByType, makeDic
from utils.databaseIo import DatabaseIo
import json
# 定义将多条数据存入数据库操作
#画ROC
drawRoc = True


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
    dbHandle = DatabaseIo()
    if not dbHandle:
        return None

    sql_dr = """select *          from course_dr"""
    sql_course = "select id , system_course_id ,course_name from course_info"
    sql_user = """select user_id from user_basic_info"""

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
    #用dcL删除某些结果里没有的课程
    dc = pd.read_csv("roc/toGcn2.csv")
    dc.drop_duplicates(subset=['cid'],keep='first',inplace=True)
    dcL=list(dc['cid'])

    #保存二部图输入
    myfile = codecs.open("gcn/bg_input.csv", mode="w", encoding='utf-8')
    data.drop_duplicates(subset=[0,1],keep='first',inplace=True)
    data.reset_index(inplace=True)
    #删除结果里没有的课程
    for i in range(len(data)):
        if data[1][i] not in dcL:
            data.drop(i,axis=0,inplace=True)

    #打印去重的看有多少课程。0用户，1课程
    # aa=data.drop_duplicates(subset=[1],keep='first')
    # print(aa.reset_index())
    data.drop('index', axis=1, inplace=True)
    datalist = data.values.tolist()

    for row in datalist:
        row[0] = user_mdicr[int(row[0])-1]
        myfile.write(str(row[0]))
        myfile.write(",")
        myfile.write(str(row[1]))
        myfile.write(",")
        myfile.write(str(int((row[2]))))
        myfile.write("\n")
    myfile.close()

    #构造二部图的输入数据的真实值矩阵realGraph,以及长id到短id(0开始)的字典
    userListNew = list(data[0])
    for i in range(len(userListNew)):
        userListNew[i] = user_mdicr[userListNew[i]-1]
    userListNewTemp = sorted(list(set(userListNew)))
    courseListNew = list(data[1])
    courseListNewTemp = sorted(list(set(courseListNew)))
    course_mdic_new={}
    user_mdic_new={}

    for i in range(len(userListNewTemp)):
        user_mdic_new[userListNewTemp[i]] = i
    for i in range(len(courseListNewTemp)):
        course_mdic_new[courseListNewTemp[i]] = i

    realGraph = nm.zeros((len(courseListNewTemp),len(userListNewTemp)))
    print(len(courseListNewTemp))
    for i in range(len(data)):
        realGraph[[course_mdic_new[courseListNew[i]]],user_mdic_new[userListNew[i]]]=1

    nm.savetxt('roc/realGraph.txt',realGraph)
    nm.save('user_mdic_new.npy', user_mdic_new)
    nm.save('course_mdic_new.npy', course_mdic_new)

    return data, learned, course_mdic, course_mdicr, user_mdic, user_mdicr,\
        dr_length, course_length, user_length, courseList

#不切分的完整数据，用于计算二部图输出
def makeTrainMatrix(data, course_length, user_length, dr_length, course_mdic):
    all_rated_graph = nm.zeros([course_length, user_length])    # 创建所有已评价矩阵
    train_graph = nm.zeros([course_length, user_length])    # 创建训练图矩阵
    test_graph = nm.zeros([course_length, user_length])    # 创建测试图矩阵
    train_rated_graph = nm.zeros([course_length, user_length])    # 创建训练集里已评价矩阵
    testIDs = random.sample(range(1, dr_length), int(dr_length / 10))

    for index, row in data.iterrows():
            test_graph[course_mdic[row[1]], int(row[0]) - 1] = 1
            train_rated_graph[course_mdic[row[1]], int(row[0]) - 1] = 1
            all_rated_graph[course_mdic[row[1]], int(row[0]) - 1] = 1

            if (int(row[2]) >= 3.0):
                train_rated_graph[course_mdic[row[1]], int(row[0]) - 1] = row[2]
                train_graph[course_mdic[row[1]], int(row[0]) - 1] = 1

    return all_rated_graph, train_graph, test_graph, train_rated_graph

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

#后缀为_split的均为供二部图画ROC的变量，用这些构造locate作为预测值画图
def doBigraph():
    data, learned, course_mdic, course_mdicr, \
        user_mdic, user_mdicr, dr_length, course_length, \
        user_length, courseList = dataPreprocessiong()

    all_rated_graph, train_graph, test_graph, train_rated_graph = \
        makeTrainMatrix(data, course_length, user_length,
                        dr_length, course_mdic)
    all_rated_graph_split, train_graph_split, test_graph_split, train_rated_graph_split = \
        makeTrainMatrixSplit(data, course_length, user_length,
                        dr_length, course_mdic)
    # 为资源配置矩阵做准备
    kjs = nm.zeros([course_length])
    kls = nm.zeros([user_length])

    kjs_split = nm.zeros([course_length])
    kls_split = nm.zeros([user_length])

    # 求产品的度
    for rid in range(course_length):
        kjs[rid] = train_graph[rid, :].sum()
        kjs_split[rid] = train_graph_split[rid, :].sum()

    # 求用户的度
    for cid in range(user_length):
        kls[cid] = train_graph[:, cid].sum()
        kls_split[cid] = train_graph_split[:, cid].sum()

    # 计算每个用户未选择产品的度
    s = nm.ones(user_length)
    s *= course_length
    ls = s - kls

    # 为防止之后的除法出现0，手动将其改为极大值
    for i in range(course_length):
        if (kjs[i] == 0.0):
            kjs[i] = 99999
        if (kjs_split[i] == 0.0):
            kjs_split[i] = 99999
    for i in range(user_length):
        if (kls[i] == 0.0):
            kls[i] = 99999
        if (kls_split[i] == 0.0):
            kls_split[i] = 99999

    # 求资源配额矩阵
    weights = nm.zeros([course_length, course_length])
    weights_split = nm.zeros([course_length, course_length])
    # 转换为矩阵乘法和向量除法
    # 设定若干中间值
    gt = train_graph.T
    temp = nm.zeros([user_length, course_length])

    gt_split = train_graph_split.T
    temp_split = nm.zeros([user_length, course_length])
    for i in range(course_length):
        temp[:, i] = gt[:, i] / kls
        temp_split[:, i] = gt_split[:, i] / kls_split

    # temp = nm.array(sparkMultiply(train_graph, temp,
    #                              user_length, course_length))
    temp = nm.matmul(train_graph, temp)
    temp_split = nm.matmul(train_graph_split, temp_split)
    for i in range(course_length):
        weights[i, :] = temp[i, :] / kjs
        weights_split[i, :] = temp_split[i, :] / kjs_split

    # 求各个用户的资源分配矩阵
    locate = nm.matmul(weights, train_rated_graph)
    locate_split = nm.matmul(weights_split, train_rated_graph_split)
    # locate = nm.array(sparkMultiply(weights, train_rated_graph,
    #                                course_length, user_length))
    # 将算法产生的推荐结果以列表形式存储

    #保存矩阵、字典，用于画ROC曲线
    if drawRoc:
        nm.savetxt("roc/bgLocate.txt", locate_split , delimiter=',')
        nm.savetxt("roc/test_graph.txt", test_graph_split, delimiter=',')
        nm.save('user_mdicr.npy', user_mdicr)
        nm.save('course_mdicr.npy', course_mdicr)
        nm.save('user_mdic.npy', user_mdic)
        nm.save('course_mdic.npy', course_mdic)



    recommend = []
    for i in range(len(locate)):
        for j in range(len(locate[i])):
            # 过滤掉用户已学习过的课程
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

#二部图原来的正常输出
def storeData(recommend_result):

    myfile = codecs.open("data.csv", mode="w", encoding='utf-8')
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

#转化为可输入GCN的数据
def storeDataAsGCNInput(recommend_result):
    myfile = codecs.open("gcn/toGcn.csv", mode="w", encoding='utf-8')
    result_data = sorted(tuple(recommend_result))
    df = pd.DataFrame(result_data)
    df = df.drop(2,axis=1)
    df.sort_values([0,3],ascending = [1,0],inplace=True)
    #df = df.groupby(0).head(5)
    # aa = df.drop_duplicates(subset=[1], keep='first')
    # print(aa.reset_index())
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

def bigraphMain():
    locate, recommend_result, learned, \
        user_length, ls, test_graph = doBigraph()
    storeData(recommend_result)
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

    # 通过矩阵对应元素相乘得到测试集的排名数据
    # 为方便后续处理，对结果进行转置
    testIndice = (indiceLocate * test_graph).T
    # 求精确度的值
    usum = 0
    # 计算测试集中每部已评分电影的距离，并求均值
    for i in range(user_length):
        if (test_graph[:, i].sum() > 0):
            usum += ((testIndice[i]).sum() / (ls[i] * test_graph[:, i].sum()))
    print("the average value of r is:")
    print(usum / user_length)

    return learned, result_data, test_graph


def Main():
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

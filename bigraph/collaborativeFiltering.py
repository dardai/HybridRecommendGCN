# -*- coding: utf-8 -*-
# 周洋涛-2019.9
# 本代码实现了协同过滤算法，与二部图算法进行比较
from __future__ import division
import numpy
from numpy import *
import csv
import time
#from texttable import Texttable
import random
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import courseRecommend as cR
from utils.extends import formatDataByType, makeDic
from globalConst import DataBaseOperateType, SetType
from utils.databaseIo import DatabaseIo
import codecs


class CF:
    def __init__(self, movies, ratings, course_length, user_length, k=10, n=20):
        self.movies = movies
        self.ratings = ratings
        self.course_length = course_length
        self.user_length = user_length
        # 邻居个数
        self.k = k
        # 推荐个数
        self.n = n
        # 训练数据中用户对电影的评分
        # 数据格式{'UserID：用户ID':[(MovieID：电影ID,Rating：用户对电影的评星)]}
        self.userDict = {}
        # 训练数据中对某电影评分的用户
        # {'1',[1,2,3..],...}
        self.ItemUser = {}
        # 数据格式：{'MovieID：电影ID',[UserID：用户ID]}
        # 邻居的信息
        self.neighbors = []
        # 推荐列表
        self.recommandList = []
        self.cost = 0.0

        # 训练数据与测试数据
        self.trans_data = []
        self.test_data = []
        # 测试数据中用户对电影的评分
        self.test_userDict = {}
        # 测试数据中对某电影评分的用户
        self.test_ItemUser = {}

        self.test_neighbors = []
        # 推荐列表
        self.test_recommandList = []
        self.trans_sort = []

        self.trans_data_id = []
        self.test_userid = []

        self.data = []
        self.data_id = []

        self.result = []
        # 统计用户未选择的课程数
        self.ls = {}
        # 记录推荐列表中的课程数
        self.recommandItem = {}
        self.Nd = 0.0
        # 记录测试矩阵和预测矩阵
        self.testMatrix = numpy.zeros([course_length, user_length])
        self.preMatrix = numpy.zeros([course_length, user_length])

    #  增加划分函数
    #  对数据进行划分，取10000个数据作为训练集，剩下数据作为测试集
    def get_sample(self):

        testIDs = random.sample(list(arange(1, len(self.ratings))), int(9 * len(self.ratings) / 10))
        for i in range(len(self.ratings)):
            if i not in testIDs:
                self.testMatrix[self.ratings[i][1]][self.ratings[i][0]] = 1
            # print(i)
            if self.ratings[i][2] >= 3.0:
                self.data.append(self.ratings[i])
                if self.ratings[i][0] not in self.data_id:
                    self.data_id.append((self.ratings[i][0]))
                if i in testIDs:
                    # self.transGraph[int(self.ratings[i][1]),int(self.ratings[i][0])] = 1
                    self.trans_data.append(self.ratings[i])
                    self.trans_data_id.append((self.ratings[i][0]))
                else:
                    # self.testGraph[int(self.ratings[i][1]),int(self.ratings[i][0])] = 1
                    self.test_userid.append((self.ratings[i][0]))
                    self.test_data.append(self.ratings[i])
        print('ratings')
        print(len(self.ratings))
        print(len(self.trans_data))
        print(len(self.test_data))

    # 基于用户的推荐
    # 根据对电影的评分计算用户之间的相似度
    def recommendByUser(self):
        self.get_sample()
        self.formatRate()
        print(self.data_id)
        print(len(self.data_id))
        self.wirtedata()
        for row in self.data_id:
            userId = row
            self.neighbors = []
            # 推荐个数 等于 本身评分电影个数，用户计算准确率
            #  self.n = len(self.userDict[userId])
            self.neighbors = self.getNearestNeighbor(userId)
            data = self.getrecommandList(userId)
            self.recommandList = sorted(self.recommandList, key=lambda x: x[2], reverse=True)
            self.result.extend(self.recommandList)
        self.storeData()
        self.test_compare()
        self.makeROC()

    # 获取推荐列表
    def getrecommandList(self, userId):
        temp_recommandList = []
        self.recommandList = []
        # 建立推荐字典
        recommandDict = {}
        for neighbor in self.neighbors:
            courses = self.userDict[neighbor[1]]
            for course in courses:
                if (course[0] in recommandDict):
                    recommandDict[course[0]] += neighbor[0]
                else:
                    recommandDict[course[0]] = neighbor[0]

        # 建立推荐列表
        for key in recommandDict:
            self.recommandList.append([userId, key, recommandDict[key]/self.user_length])
            self.preMatrix[key][userId] = recommandDict[key]/self.user_length
            if (key not in self.recommandItem):
                if (recommandDict[key]/self.user_length) > 0.0:
                    self.recommandItem[key] = 1
                    self.Nd += 1
        temp_recommandList = self.recommandList
        return temp_recommandList

    def makeROC(self):
        # locate矩阵的标准化
        lmax = self.preMatrix.max()
        regLocate = self.preMatrix / (lmax + 0.1)
        #print(lmax)
        #print("-----------------------------------")
        #print(regLocate)
        # 准备数据
        flatTest = self.testMatrix.flatten()
        flatRegl = regLocate.flatten()

        fpr, tpr, threshold = roc_curve(flatTest, flatRegl)
        roc_auc = auc(fpr, tpr)

        # 开始输出ROC曲线
        lw = 2
        plt.figure(figsize=(8, 5))
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1])
        plt.ylim([0.0, 1.1])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC for collaborativeFiltering RecSys')
        plt.legend(loc="lower right")
        plt.show()

    def test_compare(self):
        # 计算预测覆盖率
        COVp = self.Nd / self.course_length
        print(self.Nd)
        print(self.course_length)
        print("prediction coverage:")
        print(COVp)
        # M统计测试数据用户数量
        M = 0.0
        # rs计算每个用户的相对排名数值
        rs = 0.0
        for i in self.test_userid:
            count = 0.0

            for j in self.result:
                if i == j[0]:
                    count = count + 1

            R = (1+count)*count/2
            rs = rs + R/self.ls[i]
            M = M + 1
        # r表示所有用户的平均相对排名
        #r = rs / M
        r = rs / self.user_length
        print("r:")
        print(r)

    '''
    将训练集数据转换为userDict和ItemUser
    '''

    def formatRate(self):
        self.userDict = {}
        self.ItemUser = {}
        #for i in self.trans_data:
        # 将ratings替换为训练集
        for i in self.data:
            # 评分最高为5 除以5 进行数据归一化
            temp = (i[1], float(float(i[2]) / 5))
            # 计算userDict {'1':[(1,5),(2,5)...],'2':[...]...}
            if (i[0] in self.userDict):
                self.userDict[i[0]].append(temp)
            else:
                self.userDict[i[0]] = [temp]
            # 计算ItemUser {'1',[1,2,3..],...}
            if (i[1] in self.ItemUser):
                self.ItemUser[i[1]].append(i[0])
            else:
                self.ItemUser[i[1]] = [i[0]]
        # 计算每一个用户未选择的课程数量
        for i in self.data_id:
            self.ls[i] = self.course_length - len(self.userDict[i])


    # 找到某用户的相邻用户
    def getNearestNeighbor(self, userId):
        neighbors = []
        temp_neighbors = []  # dist，用户id
        # 获取userId评分的电影都有那些用户也评过分
        for i in self.userDict[userId]:
            for j in self.ItemUser[i[0]]:
                if (j != userId and j not in neighbors):
                    neighbors.append(j)  # 用户id
        # 计算这些用户与userId的相似度并排序
        for i in neighbors:
            dist = self.getCost(userId, i)
            temp_neighbors.append([dist, i])  # 用户id，[dist，用户id]
        # 排序默认是升序，reverse=True表示降序
        temp_neighbors.sort(reverse=True)
        temp_neighbors = temp_neighbors[:self.k]
        return temp_neighbors

    # 格式化userDict数据
    def formatuserDict(self, userId, l):
        user = {}
        for i in self.userDict[userId]:
            user[i[0]] = [i[1], 0]  # {'电影id':[电影的评分，0]}
        for j in self.userDict[l]:
            if (j[0] not in user):
                user[j[0]] = [0, j[1]]  # 若l用户的电影没有在user中，则{'电影id':[0，l用户的评分]}
            else:
                user[j[0]][1] = j[1]  # 若l用户的电影在user中，则{'电影id':[电影的评分，l用户的评分]}
        return user

    # 计算余弦距离
    def getCost(self, userId, l):
        # 获取用户userId和l评分电影的并集
        # {'电影ID'：[userId的评分，l的评分]} 没有评分为0
        user = self.formatuserDict(userId, l)
        x = 0.0
        y = 0.0
        z = 0.0
        for k, v in user.items():
            x += float(v[0]) * float(v[0])
            y += float(v[1]) * float(v[1])
            z += float(v[0]) * float(v[1])
        if (z == 0.0):
            return 0
        return z / sqrt(x * y)

    def wirtedata(self):
        myfile = codecs.open("file_saved/colldata.txt", mode="w", encoding='utf-8')
        result_data = sorted(tuple(self.recommandList))
        myfile.write("user_id")
        myfile.write("   ")
        myfile.write("course_id")
        myfile.write("   ")
        myfile.write("recommend_value")
        myfile.write("\n")
        myfile.close()

    def storeData(self):
        myfile = codecs.open("file_saved/colldata.txt", mode="a", encoding='utf-8')
        result_data = sorted(tuple(self.result))
        for row in result_data:
            myfile.write(str(row[0]))
            myfile.write(" ")
            myfile.write(str(row[1]))
            myfile.write(" ")
            myfile.write(str(row[2]))
            myfile.write("\n")

        myfile.close()

# -------------------------开始-------------------------------
def coll_main():
    start = time.clock()
    # 获取数据
    result_dr, result_course, result_user = cR.getDataFromDB()

    # 把从course_dr中读取出来的数据以列表形式存储
    k = list()
    k = formatDataByType(SetType.SetType_List, result_dr)
    # 按course_id升序排序
    result_list = sorted(k, key=lambda z: z[1])
    # 读取user的id
    # 把从user_basic_info中读取出来的数据以列表形式存储
    user_basic_info_list = formatDataByType(SetType.SetType_Set, result_user)

    # 把从course_info中读取出来的数据以列表形式存储
    course_info_list = formatDataByType(SetType.SetType_List, result_course)

    course_length = len(result_course)
    user_length = len(result_user)
    range_length = len(result_dr)
    movies = result_course

    # 建立字典，实现课程id和索引序号之间的映射，方便后续工作
    course_mdic, course_mdicr = makeDic(course_info_list)

    # 建立字典，实现用户id和索引序号之间的映射，方便后续工作
    user_mdic, user_mdicr = makeDic(user_basic_info_list)

    ratings = list()
    for j in range(range_length):
        w = []
        w.append(user_mdic[k[j][0]])
        w.append(course_mdic[k[j][1]])
        w.append(5 * k[j][2])
        ratings.append(w)
    #print('ratings')
    #print(len(ratings))

    demo = CF(movies, ratings, course_length, user_length, k=10)
    demo.recommendByUser()
    recommend_result = demo.recommandList
    print("训练集的数据为%d条" % (len(demo.trans_data)))
    print("测试集的数据为%d条" % (len(demo.test_data)))
    end = time.clock()
    print("耗费时间： %f s" % (end - start))
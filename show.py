# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

from bigraph.courseRecommend import getDataFromDB,dataPreprocessiong,get_keys
from pandas.core.frame import DataFrame
import pandas as pd
from bigraph import *
import prettytable as pt
def Main():
    # 显示所有列
    pd.set_option('display.max_columns', None)
    # 显示所有行
    pd.set_option('display.max_rows', None)
    # 设置value的显示长度为100，默认为50
    pd.set_option('max_colwidth', 100)
    data, learned, course_mdic, course_mdicr, \
    user_mdic, user_mdicr, dr_length, course_length, \
    user_length, courseList = dataPreprocessiong()
    gcn_result = pd.read_csv('C:/Users/Administrator/Desktop/HybridRecommendGCN/gcn/resultToRoc.csv',encoding='unicode_escape')
    gcn_uid = gcn_result.uid.tolist()
    gcn_cid = gcn_result.cid.tolist()
    gcn_score = gcn_result.score.tolist()
    gcn_result = gcn_result.values.tolist()
    length = len(gcn_result)
    #print(gcn_score)
    result_data = list()
    for i in range(length):
        temp = []
        temp.append(gcn_uid[i])
        temp.append(gcn_cid[i])
        temp.append(gcn_score[i])
        temp.append(get_keys(gcn_cid[i],courseList))
        result_data.append(temp)
    result_data = sorted(result_data, key=lambda x: x[2], reverse=True)
    #print(result_data)
    while True:
        p = "请输入用户id："
        user_id = input(p.decode('utf-8').encode('gbk'))
        if user_id == "quit":
            break
        else:
            ta = pt.PrettyTable(encoding=sys.stdout.encoding)
            ta.field_names = ["User_id", "Course_id", "Course_name"]
            l1 = []
            for row in learned:
                if row[0] == int(user_id):
                    ta.add_row(row)
                    l1.append(row)
            p = "该用户已学习过的课程有："
            print p.decode('utf-8').encode('gbk')
            print("  ")
            ta.padding_width = 5
            ta.align = "l"
            ta.border = False
            print(ta)
            print("  ")
            tb = pt.PrettyTable(encoding=sys.stdout.encoding)
            tb.field_names = ["User_id", "Course_id","Recommend_value",
                              "Course_name"]
            for row in result_data:
                if int(row[0]) == int(user_id):

                    tb.add_row(row)
            p = "为该用户推荐学习的课程有："
            print p.decode('utf-8').encode('gbk')
            print("  ")
            tb.padding_width = 5
            tb.border = False
            tb.align = "l"
            print(tb)
            print("  ")

Main()
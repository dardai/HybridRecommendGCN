# -*- coding: utf-8 -*-
from utils.databaseIo import DatabaseIo
from globalConst import DataBaseOperateType, SetType, DataBaseQuery
import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
import logging

# np.set_printoptions(suppress=True, threshold=np.nan)
pd.set_option('float_format', lambda x: '%.3f' % x)
import sys

reload(sys)
sys.setdefaultencoding('utf-8')

# 按值对字典进行降序排序，输出列表结果
from utils.extends import formatDataByType, makeDic


def sort_by_value(dic):
    # logging.warning(u"运行日志：按值对字典进行降序排序并返回列表结果")
    return sorted(dic.items(), key=lambda k: k[1], reverse=True)


# 按键对字典进行升序排序，输出列表结果
def sort_by_key(dic):
    # logging.warning(u"运行日志：按键对字典进行升序排序并返回列表结果")
    return sorted(dic.items(), key=lambda k: k[0])


# 获取用户与课程的交互数据，输出元组结果（用户id，课程id，点击次数，评分）
def get_user_course():
    # logging.warning(u"运行日志：获取用户与课程的交互数据")
    dbHandle = DatabaseIo()
    if not dbHandle:
        return None

    # sql_user_course = "select account_id, course_id, click_times, score from account_course5000"
    sql_user_course = DataBaseQuery["user_course"]
    result_user_course = dbHandle.doSql(execType=DataBaseOperateType.SearchMany,
                                        sql=sql_user_course)

    # sql_user_course_changed = "select account_id, course_id, click_times, score from account_course5000"
    sql_user_course_changed = DataBaseQuery["user_course_changed"]
    result_user_course_changed = dbHandle.doSql(execType=DataBaseOperateType.SearchMany,
                                                sql=sql_user_course_changed)
    dbHandle.changeCloseFlag()

    result_user_course = list(result_user_course) + list(result_user_course_changed)

    return result_user_course


# 获取所有用户的id
def get_all_users():
    # logging.warning(u"运行日志：获取所有用户的id")
    dbHandle = DatabaseIo()
    if not dbHandle:
        return None

    # sql_user = "select id from account5000"
    sql_user = DataBaseQuery["user_id"]
    result_user = dbHandle.doSql(execType=DataBaseOperateType.SearchMany,
                                 sql=sql_user)
    dbHandle.changeCloseFlag()
    return result_user


# 根据点击次数，计算每个课程的被点击总次数，获取热门课程列表
def popular_courses():
    logging.warning(u"运行日志：获取热门课程列表")
    courses = get_user_course()
    course_click_times = {}
    for row in courses:
        # if int(row[1]) not in course_click_times:
        #     course_click_times[int(row[1])] = int(row[2])
        if row[1] not in course_click_times:
            course_click_times[row[1]] = int(row[2])
        else:
            # course_click_times[int(row[1])] = course_click_times[int(row[1])] + int(row[2])
            course_click_times[row[1]] = course_click_times[row[1]] + int(row[2])
    result = sort_by_value(course_click_times)
    result_dataframe = DataFrame(result)
    # result_dataframe.to_csv('popular.csv', index=None, header=None)
    result_dataframe.to_csv('file_saved/popular.csv', index=None, header=None)
    return result


# 根据课程评分，计算每个课程的平均评分，获取高评分课程列表
def high_score_courses():
    logging.warning(u"运行日志：获取高评分课程列表")
    courses = get_user_course()
    course_score_sum = {}
    for row in courses:
        if int(row[1]) not in course_score_sum:
            # course_score_sum[int(row[1])] = [int(row[3]), 1]
            course_score_sum[row[1]] = [int(row[3]), 1]
        else:
            # course_score_sum[int(row[1])] = [(course_score_sum[int(row[1])][0] + int(row[3])), (course_score_sum[int(row[1])][1] + 1)]
            course_score_sum[row[1]] = [(course_score_sum[row[1]][0] + int(row[3])), (course_score_sum[row[1]][1] + 1)]
    course_score = {}
    for key in course_score_sum:
        course_score[key] = course_score_sum[key][0] / course_score_sum[key][1]
    result = sort_by_value(course_score)
    result_dataframe = DataFrame(result)
    # result_dataframe.to_csv('highScore.csv', index=None, header=None)
    result_dataframe.to_csv('file_saved/highScore.csv', index=None, header=None)
    return result


# 获取每个用户的交互课程数目，计算非个性化课程比例
def get_user_rated_num():
    logging.warning(u"运行日志：计算非个性化课程比例")
    courses = get_user_course()
    user_course_num = {}
    for row in courses:
        # if int(row[0]) not in user_course_num:
        #     user_course_num[int(row[0])] = 1
        if row[0] not in user_course_num:
            user_course_num[row[0]] = 1
        else:
            # user_course_num[int(row[0])] = user_course_num[int(row[0])] + 1
            user_course_num[row[0]] = user_course_num[row[0]] + 1
    user_percentage = {}
    for key in user_course_num:
        if user_course_num[key] < 20 and user_course_num[key] >= 0:
            user_percentage[key] = float((20.0 - float(user_course_num[key])) / 20.0)
        else:
            user_percentage[key] = 0.0
    # result = sort_by_key(user_course_num)
    percentage_result = sort_by_key(user_percentage)
    return percentage_result


"""
    :return (user_id , [popular_course_num, high_score_course_num, personal_recommend_num])
"""


# 获取每个用户的推荐热门课程、高评分课程和个性化课程对应的数量
def get_course_num(y):
    logging.warning(u"运行日志：获取每个用户的推荐热门课程、高评分课程和个性化课程对应的数量")
    users = get_all_users()
    percentage = get_user_rated_num()
    course_num = {}
    for row in percentage:
        # course_num[int(row[0])] = [int(y * float(row[1]) / 2), int(y * float(row[1]) / 2), (y - int(y * float(row[1]) / 2) - int(y * float(row[1]) / 2))]
        course_num[row[0]] = [int(y * float(row[1]) / 2), int(y * float(row[1]) / 2),
                              (y - int(y * float(row[1]) / 2) - int(y * float(row[1]) / 2))]
    for row in users:
        # if int(row[0]) not in course_num:
        #     course_num[int(row[0])] = [int(y / 2), (y - y / 2), 0]
        if row[0] not in course_num:
            course_num[row[0]] = [int(y / 2), (y - y / 2), 0]
    result = sort_by_key(course_num)
    return result


def get_online_result():
    logging.warning(u"运行日志：获取在线推荐模块的推荐结果")
    # online_run()
    # online = pd.read_csv('online/online.csv', names=['uid', 'cid', 'value']).astype(str)
    online = pd.read_csv('file_saved/online.csv', names=['uid', 'cid', 'value']).astype(str)
    online_list = online.values.tolist()
    result = sorted(online_list, key=lambda x: x[2], reverse=True)
    return result


def fusion(y):
    logging.warning(u"运行日志：混合个性化推荐课程、热门课程、高评分课程")
    popular_course = popular_courses()
    high_score_course = high_score_courses()
    recommend_num = get_course_num(y)
    online_result = get_online_result()
    result = []
    for row in recommend_num:
        temp_courses = {}
        online_count = 0
        popular_count = 0
        high_score_count = 0

        for online_row in online_result:
            if online_count == row[1][2]:
                break
            # elif long(online_row[0]) == row[0]:
            elif online_row[0] == row[0]:
                # if long(online_row[1]) not in temp_courses.keys():
                if online_row[1] not in temp_courses.keys():
                    online_count = online_count + 1
                    # temp_courses[int(online_row[1])] = online_row[2]
                    temp_courses[online_row[1]] = online_row[2]

        popular_num = int((y - len(temp_courses)) / 2)
        for i in range(len(popular_course)):
            if popular_count == popular_num:
                break
            elif popular_course[i][0] not in temp_courses.keys():

                popular_count = popular_count + 1
                temp_courses[popular_course[i][0]] = -2

        high_score_num = y - len(temp_courses)
        for i in range(len(high_score_course)):
            if high_score_count == high_score_num:
                break
            elif high_score_course[i][0] not in temp_courses.keys():
                high_score_count = high_score_count + 1
                temp_courses[high_score_course[i][0]] = -1

        for key in temp_courses:
            temp = [row[0], key, temp_courses[key]]
            result.append(temp)
    result = sorted(result, key=lambda k: k[2], reverse=True)
    result_dataframe = DataFrame(result)
    # result_dataframe.to_csv('outputFusion.csv', index=None, header=None)
    result_dataframe.to_csv('file_saved/outputFusion.csv', index=None, header=None)
    return result_dataframe, popular_course


def get_course_name(value, courseList):
    # logging.warning(u"运行日志：获取课程名称")
    for row in courseList:
        if row[0] == value:
            return row[1]


def get_course_video(value, courseList):
    data = []  # 课程名称、课程简介、课程模拟视频（先用图片替代）
    for row in courseList:
        if row[0] == value:
            data.append(row[1])
            data.append(row[3])
            data.append(row[2])
            return data


def get_course_classify(value, classifyList):
    # 获取课程的类别
    for row in classifyList:
        if row[0] == value:
            return row[2]


def get_course_click_times(value, click_times):
    # 获取对应课程的点击次数
    data = 0
    for row in click_times:
        if row[0] == value:
            data = row[1]
    return data


def get_couse_info():
    # logging.warning(u"运行日志：获取课程信息")
    dbHandle = DatabaseIo()
    if not dbHandle:
        return None
    # sql_course = "select id, name from course5000"
    sql_course = DataBaseQuery["course_info"]
    result_course = dbHandle.doSql(execType=DataBaseOperateType.SearchMany,
                                   sql=sql_course)
    dbHandle.changeCloseFlag()
    courseList = formatDataByType(SetType.SetType_List, result_course)
    return courseList


def get_couse_info_with_video():
    # logging.warning(u"运行日志：获取课程信息")
    dbHandle = DatabaseIo()
    if not dbHandle:
        return None
    # 先用图片替代模拟视频
    # sql_course = "select id, name, image, description from course5000"
    sql_course = DataBaseQuery["interface_video"]
    result_course = dbHandle.doSql(execType=DataBaseOperateType.SearchMany,
                                   sql=sql_course)
    dbHandle.changeCloseFlag()
    courseList = formatDataByType(SetType.SetType_List, result_course)
    return courseList


def get_course_info_with_image():
    dbHandle = DatabaseIo()
    if not dbHandle:
        return None
    # sql_course = "select id, name, image, description from course5000"
    sql_course = DataBaseQuery["interface_image"]
    # sql_classify = "select id, course_name, classify_name from course_classify5000"
    sql_classify = DataBaseQuery["classify_info"]
    result_course = dbHandle.doSql(execType=DataBaseOperateType.SearchMany,
                                   sql=sql_course)
    result_course_classify = dbHandle.doSql(execType=DataBaseOperateType.SearchMany,
                                            sql=sql_classify)
    dbHandle.changeCloseFlag()
    courseList = formatDataByType(SetType.SetType_List, result_course)
    classifyList = formatDataByType(SetType.SetType_List, result_course_classify)
    return courseList, classifyList


def format_result(userid, y):
    logging.warning(u"运行日志：格式化混合推荐结果传递给接口")
    recommend_num = y
    result_dataframe, click_times = fusion(recommend_num)
    result_list = result_dataframe.values.tolist()
    # print(result_list)
    courseList = get_couse_info()
    data = []
    for row in result_list:
        temp_dict = {}
        if row[0] == str(userid):
            temp_dict["courseId"] = str(row[1])
            # temp_dict["courseName"] = str(get_course_name(int(row[1]), courseList))
            temp_dict["courseName"] = str(get_course_name(row[1], courseList))
            if float(row[2]) > 5:
                row[2] = 5
            temp_dict["recommendWays"] = str(row[2])
            data.append(temp_dict)
    return data


def format_result_with_video(userid, y):
    logging.warning(u"运行日志：格式化带有视频的推荐结果给接口")
    recommend_num = y
    result_dataframe, click_times = fusion(recommend_num)
    result_list = result_dataframe.values.tolist()
    courseList = get_couse_info_with_video()
    data = []
    for row in result_list:
        temp_dict = {}
        if row[0] == str(userid):
            temp_course_info = get_course_video(row[1], courseList)
            temp_dict["courseName"] = str(temp_course_info[0])
            temp_dict["courseId"] = str(row[1])
            temp_dict["description"] = str(temp_course_info[1])
            temp_dict["video"] = str(temp_course_info[2])
            if float(row[2]) > 5:
                row[2] = 5
            temp_dict["recommendWays"] = str(row[2])
            data.append(temp_dict)
    return data


def format_result_with_image(userid, y):
    logging.warning(u"运行日志：格式化带有图片的推荐结果给接口")
    recommend_num = y
    result_dataframe, click_times = fusion(recommend_num)
    result_list = result_dataframe.values.tolist()
    courseList, classifyList = get_course_info_with_image()
    data = []
    for row in result_list:
        temp_dict = {}
        if row[0] == str(userid):
            temp_course_info = get_course_video(row[1], courseList)
            temp_dict["courseName"] = str(temp_course_info[0])
            temp_dict["image"] = str(temp_course_info[2])
            temp_dict["courseClassify"] = str(get_course_classify(row[1], classifyList))
            temp_dict["clickTimes"] = str(get_course_click_times(row[1], click_times))
            temp_dict["description"] = str(temp_course_info[1])
            if float(row[2]) > 5:
                row[2] = 5
            temp_dict["recommendWays"] = str(row[2])
            data.append(temp_dict)
    return data

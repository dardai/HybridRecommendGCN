# -*- coding: utf-8 -*-

import csv
import numpy as np

from enum import Enum

from databaseIo import DatabaseIo
from globalConst import DataBaseOperateType, getEnumValue


class CourseType(Enum):
    CourseType_None = 0
    CourseType_Vedio = 1
    CourseType_RichMedia = 2
    CourseType_Text = 3
    CourseType_Scrom = 4


def transformCourseType(courseType):
    result = CourseType.CourseType_None

    if courseType.encode("utf-8") == '视频':
        result = CourseType.CourseType_Vedio
    elif courseType.encode("utf-8") == '富媒体':
        result = CourseType.CourseType_RichMedia
    elif courseType.encode("utf-8") == '文本':
        result = CourseType.CourseType_Text
    elif courseType.encode("utf-8") == 'Scrom':
        result = CourseType.CourseType_Scrom
    else:
        result = CourseType.CourseType_None

    return result


def makeFeature():
    user_feature = []
    course_feature = []

    user_dict, course_dict, user_data_dict, \
        course_data_dict = getAllUserAndCourse()

    user_length = len(user_dict)
    course_length = len(course_dict)

    for index in range(user_length):
        one_list = []
        one_list.append(user_data_dict[index][1])
        user_feature.append(one_list)
    for index in range(course_length):
        value = CourseType.CourseType_None
        if index in course_data_dict.keys():
            value = transformCourseType(course_data_dict[index][2])

        course_feature.append(getEnumValue(value))

    course_features = np.zeros((course_length, 4), dtype=np.float32)
    for index in course_dict.keys():
        other_index = course_feature[index] - 1
        course_features[index][other_index] = 1

    return user_feature, course_features


def getAllUserAndCourse():
    u_nodes, v_nodes, ratings = [], [], []

    with open('mat.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            u_nodes.append(int(row[0]))
            v_nodes.append(int(row[1]))
            ratings.append(int(row[2]))

    user_dict = {i: r for i, r in enumerate(list(set(u_nodes)))}
    course_dict = {i: r for i, r in enumerate(list(set(v_nodes)))}

    user_data_dict = getAllUserInfo(user_dict)
    course_data_dict = getAllCourseInfo(course_dict)

    return user_dict, course_dict, user_data_dict, course_data_dict


def getAllUserInfo(user_dict):
    sql = 'SELECT user_id, points, position, gender FROM user_basic_info'

    dbHandle = DatabaseIo()

    dataList = dbHandle.doSql(DataBaseOperateType.SearchMany, sql)

    user_data_dict = makeDataDict(user_dict, dataList)

    return user_data_dict


def getAllCourseInfo(course_dict):
    sql = 'SELECT id, course_differ, course_type FROM course_info'

    dbHandle = DatabaseIo()

    dataList = dbHandle.doSql(DataBaseOperateType.SearchMany, sql)

    course_data_dict = makeDataDict(course_dict, dataList)

    return course_data_dict


def makeDataDict(index_dict, data_list):
    data_dict = {}
    for index in index_dict.keys():
        for one_data in data_list:
            if index_dict[index] == one_data[0]:
                data_dict[index] = one_data
                break

    return data_dict


if __name__ == '__main__':
    print("main")
    makeFeature()

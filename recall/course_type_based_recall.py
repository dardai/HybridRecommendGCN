# -*- coding: utf-8 -*-

import csv
import numpy as np

from utils.databaseIo import DatabaseIo
from globalConst import DataBaseOperateType, getEnumValue


# 基于阈值，拿到一段时间内交互记录里的课程类别
def get_types():
    date_threshold = 1 #周期的阈值，暂定为1个月
    original_course_ids = get_original_course_id(date_threshold)  # 拿下周期内的原始课程id


    user_dict, course_dict, user_data_dict, \
        course_data_dict = getAllUserAndCourse()

    user_length = len(user_dict)
    course_length = len(course_dict)
    max_point = 1

    for key, value in user_data_dict.items():
        if max_point < value[1]:
            max_point = value[1]
    for index in range(user_length):
        one_list = []
        one_list.append(float(user_data_dict[index][1]) / max_point)
        user_feature.append(one_list)
    for index in range(course_length):
        value = CourseType.CourseType_None
        if index in course_data_dict.keys():
            value = transformCourseType(course_data_dict[index][1])

        course_feature.append(getEnumValue(value))

    course_features = np.zeros((course_length, 136), dtype=np.float32)
    for index in course_dict.keys():
        other_index = course_feature[index] - 1
        course_features[index][other_index] = 1

    #print(user_feature)
    #print('----------------')
    #print(course_features)

    return user_feature, course_features


def get_original_course_id(threshold):
    sql = 'SELECT course_id FROM account_course'

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

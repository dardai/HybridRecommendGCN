#!/usr/bin/env python 
# -*- coding:utf-8 -*-

from utils.databaseIo import DatabaseIo
from globalConst import DataBaseOperateType, DataBaseQuery
import pandas as pd
import logging 

#筛选增量数据
def getChangedData(d):

    sql_select_course = '''select account_id,course_id
            FROM account_course5000
            WHERE duration > 10
            AND UNIX_TIMESTAMP(update_time) > UNIX_TIMESTAMP('{0}')
            '''.format(d)

    dbHandle = DatabaseIo()
    if not dbHandle:
        return None

    result = dbHandle.doSql(execType=DataBaseOperateType.SearchMany,
                            sql=sql_select_course)
    df = pd.DataFrame(list(result),columns=['uid','cid'])
    return df

#同数据二部图预处理
def updateCourseDrChanged(d):
    logging.warning(u"运行日志：在线模块数据处理")

    # sql_select_course = '''select user_id,course_id,
    #     (CASE
    #     WHEN learning_time>10 THEN 1
    #     ELSE 0 END)as time,
    #     CASE collect_status
    #     WHEN 'YES' THEN 1 ELSE 0 END AS collect,
    #     CASE commit_status
    #     WHEN 'YES' THEN 1.5 ELSE 0 END AS commit_status,
    #     CASE
    #     WHEN score>50 THEN 1 ELSE 0 END AS score
    #     FROM user_course_changed
    #     WHERE learning_time > 10
    #     AND UNIX_TIMESTAMP(update_time) > UNIX_TIMESTAMP('{0}')'''.format(d)
    #
    # sql_insert_course_dr_changed = '''INSERT INTO course_dr_changed(user_id, course_id, recommend_value)
    #                     VALUES (%s, %s, %s)'''
    # sql_clean_course_dr_changed = 'truncate table course_dr_changed;'

    # sql_select_course = '''select account_id,course_id,
    #         (CASE
    #         WHEN duration>10 THEN 1
    #         ELSE 0 END)as time,
    #         CASE collect_status
    #         WHEN 'YES' THEN 1 ELSE 0 END AS collect,
    #         CASE commit_status
    #         WHEN 'YES' THEN 1.5 ELSE 0 END AS commit_status,
    #         CASE
    #         WHEN score>50 THEN 1 ELSE 0 END AS score
    #         FROM account_course5000
    #         WHERE duration > 10
    #         AND UNIX_TIMESTAMP(update_time) > UNIX_TIMESTAMP('{0}')'''.format(d)
    sql_select_course = DataBaseQuery["online_predeal_select_course"].format(d)

    sql_insert_course_dr = '''INSERT INTO course_dr5000_changed(user_id, course_index, recommend_value)
                        VALUES (%s, %s, %s)'''
    # sql_insert_course_dr = DataBaseQuery["online_predeal_insert_course_dr"]

    sql_clean_course_dr = 'truncate table course_dr5000_changed;'
    # sql_clean_course_dr = DataBaseQuery["online_clean_course_dr"]

    dbHandle = DatabaseIo()
    if not dbHandle:
        return None

    result = dbHandle.doSql(execType=DataBaseOperateType.SearchMany,
                            sql=sql_select_course)

    userCourseList = list()
    for row in result:
        user_id, course_id, time, collect, commit, score = row
        count = (3 * time + 2 * commit + collect + score) / 8
        userCourseList.append(tuple([user_id, course_id, count]))

    insertTuple = tuple(userCourseList)

    dbHandle.doSql(DataBaseOperateType.InsertOne, sql_clean_course_dr)
    dbHandle.changeCloseFlag()
    dbHandle.doSql(DataBaseOperateType.InsertMany, sql_insert_course_dr,
                   insertTuple)
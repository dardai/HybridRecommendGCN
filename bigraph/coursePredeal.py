# -*- coding: utf-8 -*-


from utils.databaseIo import DatabaseIo
from globalConst import DataBaseOperateType, DataBaseQuery
import logging


def updateCourseDr():
    logging.warning(u"运行日志：开始数据预处理")
    print("run coursePredeal...")

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
    #         FROM account_course5000'''
    sql_select_course = DataBaseQuery["predeal_user_course"]
    # sql_insert_course_dr = '''INSERT INTO course_dr(user_id, course_index, recommend_value)
    #                     VALUES (%s, %s, %s)'''
    sql_insert_course_dr = DataBaseQuery["predeal_insert_course_dr"]
    # sql_clean_course_dr = 'truncate table course_dr;'
    sql_clean_course_dr = DataBaseQuery["predeal_clean_course_dr"]

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
    print("coursePredeal success")

# def main():
#    updateCourseDr()

# main()

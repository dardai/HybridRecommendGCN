# -*- coding: utf-8 -*-
import datetime

from utils.databaseIo import DatabaseIo
from globalConst import DataBaseOperateType, getEnumValue


# 基于阈值，拿到一段时间内交互记录里的课程类别
def get_types():
    # 周期的阈值，暂定为1个月
    date_threshold = 1
    # 取出周期内的原始课程id
    original_course_ids = get_original_course_id(date_threshold)
    # 基于课程id找到课程类别id，利用type_id过滤course_id
    target_course_ids = []
    for course_id in original_course_ids:
        course_ids = get_target_course_id(str(course_id[0]))
        target_course_ids += course_ids

    return target_course_ids


    # 基于阈值设定时间取回课程id
def get_original_course_id(threshold):
    m = datetime.datetime.now().month
    if m == 1:
        m = 12
    else:
        m -= threshold
    if m < 10:
        m = '0' + str(m)
    else:
        m = str(m)
    time_line = str(datetime.datetime.now().year) + '-' + str(m)

    sql = 'SELECT DISTINCT course_id FROM account_course5000 WHERE date_format(update_time,' \
          ' \'%Y-%m-%d %H:%i:%s\') LIKE \'' + time_line + '____________\''

    dbHandle = DatabaseIo()

    course_ids = dbHandle.doSql(DataBaseOperateType.SearchMany, sql)

    return course_ids


# 基于给定课程id找到全部同类型课程id
def get_target_course_id(course_id):
    sql = 'SELECT DISTINCT id from course_classify5000 WHERE classify_id in (SELECT classify_id FROM course_classify5000 WHERE id=' + course_id + ')'

    dbHandle = DatabaseIo()

    course_ids = dbHandle.doSql(DataBaseOperateType.SearchMany, sql)

    return course_ids


# 基于目标课程id对学习记录进行召回
def get_recalled_records(target_course_ids):
    # 把目标课程id拼成set，方便后续查找
    course_id_set = "("
    for id in target_course_ids:
        course_id_set += '\''+str(id[0])+"\',"
    course_id_set = course_id_set[0:-1]
    course_id_set += ')'
    records = get_target_records(course_id_set)
    return records

# 基于目标课程id对学习记录进行召回查询
def get_target_records(course_id_set):
    sql = 'SELECT * from account_course5000 WHERE course_id in ' + course_id_set

    dbHandle = DatabaseIo()

    records = dbHandle.doSql(DataBaseOperateType.SearchMany, sql)

    return records


# 召回的主流程函数
def recall():
    course_ids = get_types()
    records = get_recalled_records(course_ids)
    return records


if __name__ == '__main__':
    print(recall())

# -*- coding: utf-8 -*-

from globalConst import SqlType

from utils.sqlMap import SqlMap
from utils.extends import printSet, getAppointedData, getAppointedDataAndScore

def resort(data, id):
    search_name_score_list = getAppointedDataAndScore(data, 'title')
    search_name_list = getAppointedData(data, 'title')
    rec = getRec(id)
    temp = []
    tempDict = {}
    for i in range(0,len(rec), 2):
        temp.append(rec[i])
    rec = temp
    same = sorted(set(search_name_list).intersection(set(rec)), key=search_name_list.index)
    remain = sorted(set(search_name_list).difference(set(rec)), key=search_name_list.index)
    rec_remain = sorted(set(rec).difference(set(search_name_list)), key=rec.index)
    rank = {}

    if same and len(same) > 0:
        for class_name in same:
            for item in search_name_score_list:
                if class_name == item['title']:
                    item['_score'] += 2
                    break

    search_name_score_list = sorted(search_name_score_list, key = lambda x: x["_score"], reverse = True)

    search_name_list = []
    for item in search_name_score_list:
        search_name_list.append(item['title'])
    for item in rec_remain:
        search_name_list.append(item)

    return search_name_list


def getRec(uid):
    sql = SqlMap[SqlType.GetRecommendClassNameByUid]
    row = doSql(1, sql, uid)

    rec = []
    for name in row:
        rec.append(name[0])

    return rec

def getScore(data, classId):
    sql = SqlMap[SqlType.GetRecommendClassScoreByCourseid]
    return doSql(2, sql, classId)

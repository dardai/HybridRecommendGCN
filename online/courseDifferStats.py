#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import pandas as pd
import logging
from utils.databaseIo import DatabaseIo
from globalConst import DataBaseOperateType, DataBaseQuery
from changedPredeal import getChangedData
from gcn.feature import transformCourseType


def differFusion(d):
    logging.warning(u"运行日志：在线模块类别特征融合")
    print ("run differFusion...")
    # sql_select_differ = '''select course_differ
    #                                 FROM course_info
    #                                 WHERE id = '{0}'
    #                                 '''
    # sql_select_differ = '''select classify_id
    #                                     FROM course_classify5000
    #                                     WHERE id = '{0}'
    #                                     '''
    sql_select_differ = DataBaseQuery["online_select_differ"]
    dbHandle = DatabaseIo()
    if not dbHandle:
        return None

    changedData = getChangedData(d)
    group = changedData.groupby(['uid'])
    grouped_cData = pd.DataFrame({'cid':group['cid'].apply(list)}).reset_index()
    gclist = grouped_cData['cid'].values.tolist()
    ulist = grouped_cData['uid'].values.tolist()
    #增量数据中每个用户所选的课程的类别列表
    uclist = []
    for i in range(len(ulist)):
        temp = []
        temp.append(ulist[i])
        for j in range(len(gclist[i])):
            result = dbHandle.doSql(execType=DataBaseOperateType.SearchMany,
                                    sql=sql_select_differ.format(gclist[i][j]))
            # gclist[i][j] = int(transformCourseType(result[0][0]))
            gclist[i][j] = int(result[0][0])
        temp.append(list(set(gclist[i])))
        uclist.append(temp)

    # print uclist
    # allData = pd.read_csv('../resultToRoc.csv')
    allData = pd.read_csv('../file_saved/resultToRoc.csv')
    auid = allData['uid'].values.tolist()
    acid = allData['cid'].values.tolist()
    avalue = allData['score'].values.tolist()
    print uclist
    # 从全量数据和增量数据中匹配交互过相同类别课程的目标用户，并对该目标用户在全量数据中与增量数据中类别相匹配的课程的推荐值增加0.2
    # 如果对一个目标用户，增量数据里面有全量数据中没有交互过的类别，那不算它，只给交互过的类别增加0.2
    for i in range(len(uclist)):
        for j in range(len(auid)):
            if auid[j] == long(uclist[i][0]):
                result = dbHandle.doSql(execType=DataBaseOperateType.SearchMany,
                                        sql=sql_select_differ.format(acid[j]))
                # checkcid = int(transformCourseType(result[0][0]))
                checkcid = int(result[0][0])
                if checkcid in uclist[i][1]:
                    avalue[j] += 0.2
                    # print checkcid,acid[j],avalue[j]

    differData = pd.DataFrame({0:auid,1:acid,2:avalue})

    # differData.to_csv('differData.csv',index=None,header=None)
    differData.to_csv('../file_saved/differData.csv',index=None,header=None)
    print ("differFusion success")
    return differData
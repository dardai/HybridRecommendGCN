#!/usr/bin/env python
# -*- coding:utf-8 -*-

import pandas as pd
import logging
from utils.databaseIo import DatabaseIo
from globalConst import DataBaseOperateType, DataBaseQuery

def fusion():

    logging.warning(u"运行日志：在线模块数据融合")
    print ("run fusion...")
    # differAllData = pd.read_csv('dislikeData.csv',names=['uid', 'cid', 'score'])
    # changedData = pd.read_csv('changedBigraph.csv',names=['uid', 'cid', 'score'])
    differAllData = pd.read_csv('../file_saved/dislikeData.csv',names=['uid', 'cid', 'score'])
    changedData = pd.read_csv('../file_saved/changedBigraph.csv',names=['uid', 'cid', 'score'])

    dbHandle = DatabaseIo()
    if not dbHandle:
        return None

    #找出两个df中uid和cid对相同的行，存到mergeData中
    mergeData = pd.merge(differAllData,changedData,on=['uid','cid'],how='inner')
    print mergeData

    if not mergeData.empty:
        muid = mergeData['uid'].values.tolist()
        mcid = mergeData['cid'].values.tolist()
        #ct_list储存mergeData中对应用户-课程的点击数
        ct_list = []
        for i in range(len(muid)):
            # sql_select_click_times = '''select click_times
            #         FROM account_course5000
            #         WHERE user_id = '{0}'
            #         AND course_id = '{1}'
            #         '''
            sql_select_click_times = DataBaseQuery["online_select_click_times"]

            result = dbHandle.doSql(execType=DataBaseOperateType.SearchMany,
                                    sql=sql_select_click_times.format(muid[i],mcid[i]))
            if not result:
                ct_list.append(0)
            else:
                ct_list.append(float(result[0][0]))


        cmax = max(ct_list)
        #norm_ctlist为ct_list进行规范化后的结果，作为惩罚因子加到对应的推荐值上
        norm_ctlist = []
        for i in ct_list:
            norm_ctlist.append(1-(i/(cmax+1)))

        #dvalue融合了课程类别统计的推荐值，cvalue是增量数据进行二部图运算得到的推荐值
        # fvalue为dvalue、cvalue和惩罚因子norm_ctlist融合后的推荐值
        dvalue = mergeData['score_x'].values.tolist()
        cvalue = mergeData['score_y'].values.tolist()
        fvalue = map(lambda(a,b,c):a*0.3+b+c,zip(cvalue,dvalue,norm_ctlist))

        #mergeData增量数据变为进行融合后的结果集
        mergeData.drop(['score_x','score_y'],axis=1,inplace=True)
        mergeData['score'] = fvalue

        #fusionData为增量数据和全量数据的融合数据
        #先在融合了类别因素的全量数据中添加mergeData，就会出现重复的用户-课程对，利用drop_duplicates去重并保留添加的mergeData数据
        fusionData = differAllData.append(mergeData)
        fusionData.drop_duplicates(subset=['uid','cid'],keep='last',inplace=True)
        fusionData.sort_values(by=['uid'],ascending=True,inplace=True)
        fusionData.reset_index(inplace=True)
        fusionData.drop(['index'],axis=1,inplace=True)
        print ("fusion success")
    else:
        #增量数据与离线全量数据中没有相同的用户-课程对，不进行融合，只根据类别加分
        fusionData = differAllData
        print ("no changed data..")


    # baseData = pd.read_csv('../resultToRoc.csv')
    baseData = pd.read_csv('../file_saved/resultToRoc.csv')
    updatedBaseData = baseData.append(fusionData)
    updatedBaseData.drop_duplicates(subset=['uid','cid'],keep='last',inplace=True)
    # updatedBaseData.to_csv('../resultToRoc.csv',index=None)
    updatedBaseData.to_csv('../file_saved/resultToRoc.csv',index=None)
    # updatedBaseData.to_csv('online.csv', header=None, index=None)
    updatedBaseData.to_csv('../file_saved/online.csv', header=None, index=None)

    return fusionData
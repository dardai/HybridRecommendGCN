# -*- coding: utf-8 -*-

from utils.globalConst import SetType

def formatDataByType(setType, data):
    result = None
    if setType == SetType_List:
        result = list()
    elif setType == SetType_Set:
        result = set()
    else:
        pass
    for temp in data:
        result.append(list(temp))

    return result

def makeDic(dataList):
    index = 0
    mdic, mdicr = {}, {}
    for row in dataList:
        mdic[int(row[0])] = index
        mdicr[index] = int(row[0])
        index += 1

    return mdic, mdicr

'''
    打印集合(广义的集合)
    其实可以打印所有类型的数据, 用于调试的
'''
def printSet(data, depth = 0):
    tab_length = depth * 4
    begin = '\t'
    end = '\t'
    if isinstance(data, set) or isinstance(data, dict):
        begin = begin.__add__('{').expandtabs(tab_length)
        end = end.__add__('},').expandtabs(tab_length)
    elif isinstance(data, list):
        begin = begin.__add__('[').expandtabs(tab_length)
        end = end.__add__('],').expandtabs(tab_length)
    elif isinstance(data, tuple):
        begin = begin.__add__('(').expandtabs(tab_length)
        end = end.__add__('),').expandtabs(tab_length)
    else:
        print('{},'.format(data))
        return

    print(begin)
    if isinstance(data, dict):
        for key, value in data.items():
            if isSet(value):
                printSet(value, depth+1)
                continue
            print(('\t{} : {},'.format(key, value)).expandtabs(tab_length+4))
    if isinstance(data, set) or isinstance(data, list) or isinstance(data, tuple):
        for item in data:
            if isSet(item):
                printSet(item, depth+1)
                continue
            print('\t{},'.format(item).expandtabs(tab_length+4))
    print(end)

def isSet(data):
    if isinstance(data, set) or isinstance(data, dict) or isinstance(data, list) or isinstance(data, tuple):
        return True
    return False

# 获取搜索结果中指定字段的数据, 并将结果按列表返回
def getAppointedData(data, aList):
    filterData = []
    for source in data["hits"]["hits"]:
        temp_dict = {}
        if isinstance(aList, str):
            if source["_source"][aList]:
                filterData.append(source["_source"][aList])
        else:
            for index in aList:
                if source["_source"][index]:
                    temp_dict[index] = source["_source"][index]
            filterData.append(temp_dict)

    return filterData

def getAppointedDataAndScore(data, aList):
    filterData = []
    for source in data["hits"]["hits"]:

        temp_dict = {}

        if source["_source"][aList]:
            temp_dict[aList] = source["_source"][aList]
        else:
            for index in aList:
                if source["_source"][index]:
                    temp_dict[index] = source["_source"][index]

        temp_dict['_score'] = source["_score"]
        filterData.append(temp_dict)

    return filterData



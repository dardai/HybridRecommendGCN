# -*- coding: utf-8 -*-

from enum import Enum


class DataBaseOperateType(Enum):
    InsertOne = 1
    InsertMany = 2
    SearchOne = 3
    SearchMany = 4


class SetType(Enum):
    SetType_Set = 1
    SetType_List = 2


class ESOperateErrorCode(Enum):
    Success = 1
    NoESObject = 2
    ParamTypeError = 3
    IndexExist = 4
    DocIdExist = 5
    InsertError = 6
    DocIdNotExist = 7
    NoneIndex = 8
    ParamError = 9
    Faild = 10


class ESBodyType(Enum):
    Create = 1
    SearchOneWord = 2
    SearchWords = 3
    Suggester = 4
    SearchDocByCid = 5
    SearchAll = 6


class SqlType(Enum):
    GetRecommendClassNameByUid = 1
    GetRecommendClassScoreByCourseid = 2


def getEnumValue(enum_type):
    return enum_type.value


# -----------------   数据库配置  ---------------------
DataBaseInfo_old = {
    "address": "39.100.100.198",
    "username": "root",
    "passwd": "ASElab905",
    "basename": "learningrecommend"
}


DataBaseInfo = {
    "address": "101.133.194.114",
    "username": "train_rs",
    "passwd": "Trs123!@#",
    "basename": "train_recommended_sys"
}

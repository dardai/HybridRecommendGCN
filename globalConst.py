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
DataBaseInfo = {
    "address": "39.100.100.198",
    "username": "root",
    "passwd": "ASElab905",
    "basename": "learningrecommend"
}
 
# DataBaseInfo = {
#     "address": "101.133.194.114",
#     "username": "train_rs",
#     "passwd": "Trs123!@#",
#     "basename": "train_recommended_sys"
# }

DataBaseQuery = {
    "course_dr": "select * from course_dr5000",
    "course_info": "select id, name, classify_id from course5000",
    "user_id": "select id from account5000",
    "user_course": "select account_id, course_id, click_times, score from account_course5000",
    "user_course_changed": "select account_id, course_id, click_times, score from account_course5000",
    "interface_video": "select id, name, image, description from course5000",
    "interface_image": "select id, name, image, description from course5000",
    "classify_info": "select id, course_name, classify_name, classify_id from course_classify5000",
    "online_course_dr_changed": "select * from course_dr5000_changed",
    "online_select_differ": "select classify_id from course_classify5000 where id = '{0}'",
    "online_select_dislike": "select classify_id from course_classify where id = '{0}'",
    "online_select_click_times": "select click_times from account_course5000 where user_id = '{0}' and course_id = '{1}'",
    "online_predeal_select_course": "select account_id,course_id, (CASE WHEN duration>10 THEN 1 ELSE 0 END)as time, CASE collect_status WHEN 'YES' THEN 1 ELSE 0 END AS collect, CASE commit_status WHEN 'YES' THEN 1.5 ELSE 0 END AS commit_status, CASE WHEN score>50 THEN 1 ELSE 0 END AS score FROM account_course5000 WHERE duration > 10 AND UNIX_TIMESTAMP(update_time) > UNIX_TIMESTAMP('{0}')",
    "online_predeal_insert_course_dr": "insert into course_dr5000_changed(user_id, course_index, recommend_value) values (%s, %s, %s)",
    "online_clean_course_dr": "truncate table course_dr5000_changed;",
    "feature_classify": "select id, classify_id from course5000",
    "predeal_user_course": "select account_id,course_id, (CASE WHEN duration>10 THEN 1 ELSE 0 END)as time, CASE collect_status WHEN 'YES' THEN 1 ELSE 0 END AS collect, CASE commit_status WHEN 'YES' THEN 1.5 ELSE 0 END AS commit_status, CASE WHEN score>50 THEN 1 ELSE 0 END AS score FROM account_course5000",
    "predeal_insert_course_dr": "insert into course_dr(user_id, course_index, recommend_value) values (%s, %s, %s)",
    "predeal_clean_course_dr": "truncate table course_dr;",
    "dgl_user_info": "select id,sex from account5000",
    "popular_clean_course": "truncate table popular_course;",
    "popular_insert_course": "insert into popular_course(course_id, click_times) values (%s, %s)",
    "popular_select_course": "select course_id,click_times from popular_course ",
    "highscore_clean_course": "truncate table high_score_course;",
    "highscore_insert_course": "insert into high_score_course(course_id, score) values (%s, %s)",
    "highscore_select_course": "select course_id,score from high_score_course "

}
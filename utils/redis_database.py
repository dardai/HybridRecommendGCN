# -*- coding: utf-8 -*-
import redis

try:

    # 混合推荐结果存储的连接池，逻辑库为1
    pool_recommend = redis.ConnectionPool(
        host = "47.103.71.75",
        port = 6379,
        password = "FSLai2020#",
        db = 1
    )

    # 热门课程结果存储的连接池，逻辑库为2
    pool_popular = redis.ConnectionPool(
        host = "47.103.71.75",
        port = 6379,
        password = "FSLai2020#",
        db = 2
    )

    # 高评分课程结果存储的连接池，逻辑库为3
    pool_highscore = redis.ConnectionPool(
        host = "47.103.71.75",
        port = 6379,
        password = "FSLai2020#",
        db = 3
    )
except Exception as e:
    print(e)
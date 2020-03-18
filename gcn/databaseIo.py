# -*- coding: utf-8 -*-

import MySQLdb
from globalConst import DataBaseOperateType, DataBaseInfo


class DatabaseIo:

    def __init__(self):
        self.closeFlag = False
        self.db = MySQLdb.connect(
            DataBaseInfo["address"],
            DataBaseInfo["username"],
            DataBaseInfo["passwd"],
            DataBaseInfo["basename"],
            charset="utf8")
        if self.db:
            self.cursor = self.db.cursor()

    def changeCloseFlag(self):
        if not self.closeFlag:
            self.closeFlag = True
    '''
        执行 sql 语句
        params: execType: 执行操作的类型
                values: 执行 executemany() 所需要的值
                sql: sql 语句
                *params: 需要对 sql 语句进行格式化的参数
    '''
    def doSql(self, execType, sql, values=None, *params):
        result = None
        sql = formatSql(sql, *params)

        if not sql:
            return result

        try:
            if execType == DataBaseOperateType.InsertOne:
                result = self.cursor.execute(sql)
                self.db.commit()
            elif execType == DataBaseOperateType.InsertMany:
                result = self.cursor.executemany(sql, values)
                self.db.commit()
            elif execType == DataBaseOperateType.SearchOne:
                self.cursor.execute(sql)
                result = self.cursor.fetchone()
            elif execType == DataBaseOperateType.SearchMany:
                self.cursor.execute(sql)
                result = self.cursor.fetchall()
            else:
                pass
        except Exception:
            print("rollback()  sql = {}".format(sql))
            self.db.rollback()

        if self.closeFlag:
            self.db.close()

        return result

    '''
    def write(self, sql):
        try:
            self.cursor.execute(sql)
            self.db.commit()
        except:
            self.db.rollback()
            print('Error: unable to write data')

    def writeMany(self, sql,li):
        print('进入many')
        try:
            print('进入try')
            n = self.cursor.executemany(sql,li)
            self.db.commit()
            return n
        except Exception as e:
            traceback.print_exc()
            self.db.rollback()

    def read(self, sql):
        try:
            self.cursor.execute(sql)
            results = self.cursor.fetchall()
            return results
        except:
            print('Error: unable to fetch data')

    def close(self):
        self.db.close()
    '''


def formatSql(sql, *params):
    sql = ' '.join(sql.split())
    if sql.find('={}') == -1:
        return sql

    needFormatNumber = int((len(sql) - len(sql.replace('={}', ''))) / 3)
    if needFormatNumber > len(params):
        return None

    sql = sql.format(*params)

    return sql

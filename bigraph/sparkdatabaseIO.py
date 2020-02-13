# -*- coding: utf-8 -*-

from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.types import StructField, StringType, FloatType, StructType, \
    LongType, DoubleType, DecimalType


class SparkDBIO:
    # coding:utf-8
    def __init__(self):
        self.sc = SparkContext(appName="pyspark mysql demo")
        self.sqlContext = SQLContext(self.sc)

    # sql = """(SELECT id FROM course_dr) T"""
    # sql_dr = '(select * from course_dr) T'

    def sparkread(self, sql):
        # sc = SparkContext(appName="pyspark mysql demo")
        # sqlContext = SQLContext(sc)
        # df=sqlContext.read.format('jdbc').options(url="jdbc:mysql://localhost/learningrecommend?user=root&password=114210",dbtable=table).load().show()
        z = self.sqlContext.read.format("jdbc").option("url",
                                                  "jdbc:mysql://39.100.100.198:3306/learningrecommend?serverTimezone=GMT%2B8").option(
            "dbtable", sql).option("user", "root").option("password", "ASElab905").load()
        z.printSchema()
        # z.show()
        result = list(z.collect())
        # self.sc.stop()
        return result

    def sparksave(self, data, dtable, field):
        # sc = SparkContext(appName="pyspark mysql demo")
        # sqlContext = SQLContext(sc)
        n = self.sc.parallelize(data)
        schema = StructType(field)
        spark_df = self.sqlContext.createDataFrame(n, schema)
        spark_df.write.mode("overwrite").format("jdbc").options(
            url='jdbc:mysql://39.100.100.198:3306/learningrecommend',
            user='root',
            password='ASElab905',
            dbtable=dtable,
            # batchsize="1000",
        ).save()

    def close(self):
        self.sc.stop()

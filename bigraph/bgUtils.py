#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import codecs
import pandas as pd

#转化为可输入GCN的数据
def storeDataAsGCNInput(recommend_result):
    #修改文件目录
    #myfile = codecs.open("C:/Users/Administrator/Desktop/HybridRecommendGCN/gcn/toGcn.csv", mode="w", encoding='utf-8')
    myfile = codecs.open("gcn/toGcn.csv", mode="w", encoding='utf-8')
    result_data = sorted(tuple(recommend_result))
    df = pd.DataFrame(result_data)
    df = df.drop(2,axis=1)
    df.sort_values([0,3],ascending = [1,0],inplace=True)
    #df = df.groupby(0).head(5)
    # aa = df.drop_duplicates(subset=[1], keep='first')
    # print(aa.reset_index())
    grouped = df.values.tolist()
    #按四舍五入处理推荐值，不保存推荐值为0的数据
    for row in grouped:
        tempRow = row[2]
        if (tempRow > 5):
            tempRow = 5
        tempRow = round(tempRow)
        tempRow = int(tempRow)
        if tempRow != 0 :
            myfile.write(str(row[0]))
            myfile.write(",")
            myfile.write(str(row[1]))
            myfile.write(",")
            myfile.write(str(tempRow))
            myfile.write("\n")
    myfile.close()
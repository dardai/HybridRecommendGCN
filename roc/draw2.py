#!/usr/bin/env python 
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as nm
import pandas as pd

def drawBigraphRoc():
    #读取从二部图中保存的矩阵
    locate = nm.loadtxt('roc/bgLocate.txt', delimiter=',')
    test_graph = nm.loadtxt('roc/test_graph.txt', delimiter=',')
    #矩阵的标准化
    lmax = locate.max()
    reg_Locate = locate / (lmax + 0.1)
    #准备参数
    flatTest = test_graph.flatten()
    flatRegl = reg_Locate.flatten()

    #输入参数进行计算
    fpr, tpr, threshold = roc_curve(flatTest, flatRegl)
    roc_auc = auc(fpr, tpr)
    print("AUC=", roc_auc)
    #绘制ROC曲线
    lw = 2
    plt.figure(figsize=(8, 5))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for BiGraph')
    plt.legend(loc="lower right")
    plt.show()




def drawGCNRoc():
    test_graph = nm.loadtxt('roc/test_graph.txt', delimiter=',')
    source_data = pd.read_csv('gcn/resultToRoc.csv')

    # gcn_u_dictr = nm.load('gcn/u_dictr.npy',allow_pickle=True).item()
    # gcn_v_dictr = nm.load('gcn/v_dictr.npy',allow_pickle=True).item()
    bg_u_dict = nm.load('user_mdic.npy',allow_pickle=True).item()
    bg_v_dict = nm.load('course_mdic.npy',allow_pickle=True).item()


    datalist = source_data.values.tolist()
    dr = []
    course=[]
    user=[]
    for i in range(len(datalist)):
        user.append(datalist[i][0])
        course.append(datalist[i][1])
        dr.append(datalist[i][2])

    course = list(set(course))
    user = list(set(user))
    dr_length = len(dr)
    course_length = len(course)
    user_length = len(user)
    print(course_length,user_length)

    '''
        改uid映射
    '''
    uid = source_data.uid.tolist()
    # for i in range(len(uid)):
    #     uid[i] = gcn_u_dictr[uid[i]]
    #

    for i in range(len(uid)):
        uid[i] = bg_u_dict[uid[i]]

    '''
        改cid映射
    '''
    cid = source_data.cid.tolist()
    # for i in range(len(cid)):
    #     cid[i] = gcn_v_dictr[cid[i]]


    for i in range(len(cid)):
        cid[i] = bg_v_dict[cid[i]]

    # realList = []
    # for index, row in source_data.iterrows():
    #     if(index  in range(dr_length)):
    #        realList.append(test_graph[cid[index], uid[index]])
    # nm.savetxt("rg.txt", realList, delimiter=',')

    #标准化
    # for i in range(len(dr)):
    #     dr[i] = dr[i]/5.00
    # predict = dr

    #改551*1184
    predict=nm.zeros((551,1184))
    for i in range(len(cid)):
        predict[cid[i]][uid[i]] = dr[i]

    testGraph = test_graph.flatten()
    predGraph = predict.flatten()

    #nm.savetxt("predict.txt", predict, delimiter=',')

    #输入参数进行计算
    fpr, tpr, threshold = roc_curve(testGraph, predGraph)
    print("假阳性率：")
    print(fpr)
    print("真阳性率：")
    print (tpr)

    roc_auc = auc(fpr, tpr)
    print("AUC=", roc_auc)
    #绘制ROC曲线
    print("draw ROC(GCN)")
    lw = 2
    plt.figure(figsize=(8, 5))

    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for GCN')
    plt.legend(loc="lower right")
    plt.show()

def Main():

    drawBigraphRoc()
    drawGCNRoc()
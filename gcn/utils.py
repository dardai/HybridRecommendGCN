# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function
import csv
import numpy as np

def construct_feed_dict(placeholders, u_features, v_features, u_features_nonzero, v_features_nonzero,
                        support, support_t, labels, u_indices, v_indices, class_values,
                        dropout, u_features_side=None, v_features_side=None):
    """
    Function that creates feed dictionary when running tensorflow sessions.
    """

    feed_dict = dict()
    feed_dict.update({placeholders['u_features']: u_features})
    feed_dict.update({placeholders['v_features']: v_features})
    feed_dict.update({placeholders['u_features_nonzero']: u_features_nonzero})
    feed_dict.update({placeholders['v_features_nonzero']: v_features_nonzero})
    feed_dict.update({placeholders['support']: support})
    feed_dict.update({placeholders['support_t']: support_t})

    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['user_indices']: u_indices})
    feed_dict.update({placeholders['item_indices']: v_indices})

    feed_dict.update({placeholders['dropout']: dropout})
    feed_dict.update({placeholders['class_values']: class_values})

    if (u_features_side is not None) and (v_features_side is not None):
        feed_dict.update({placeholders['u_features_side']: u_features_side})
        feed_dict.update({placeholders['v_features_side']: v_features_side})

    return feed_dict


def print_predict_d(labels, u_indices, v_indices):
    print("-----------print_predict_d--------------")
    for user, vedio, score in zip(u_indices, v_indices, labels):
        print("{}   {}  {}".format(user, vedio, score))
    print("-----------print_predict_d--------------")


def write_csv(labels, u_indices, v_indices):
    print("-----------write_csv--------------")
    f = open('result.csv', 'wb+')
    content = csv.writer(f)
    for user, vedio, score in zip(u_indices, v_indices, labels):
        content.writerow([user, vedio, score])
    print("-----------write_csv--------------")

def write_csv2(labels, u_indices, v_indices):
    gcn_u_dictr = np.load('u_dictr.npy', allow_pickle=True).item()
    gcn_v_dictr = np.load('v_dictr.npy', allow_pickle=True).item()
    print("-----------write_csv2--------------")
    f = open('gcn/resultToRoc.csv', 'wb+')
    # f = open('C:/Users/Administrator/Desktop/HybridRecommendGCN/gcn/resultToRoc.csv', 'wb+')
    content = csv.writer(f)
    content.writerow(['uid', 'cid', 'score'])
    for user, vedio, score in zip(u_indices, v_indices, labels):
        user = gcn_u_dictr[user]
        vedio = gcn_v_dictr[vedio]
        content.writerow([user, vedio, score])
    print("-----------write_csv2--------------")

def getReversalDict(idDict):
    newDict = dict()

    for index, realId in idDict.items():
        newDict[realId] = index

    return newDict


def getRealId(idDict, idList):
    newIdList = list()

    for index in idList:
        if idDict[index]:
            newIdList.append(idDict[index])
        else:
            print("index  not   exit")

    return newIdList

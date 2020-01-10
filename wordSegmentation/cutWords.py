# -*- coding: utf-8 -*-

import jieba

from utils.extends import printSet
from searchEngine.resort import getRec

WordsFrenquencyDict = {}
StopWord = []

def wordCut(id):
    recList = getRec(id)

    for title in recList:
        tempList = jieba.lcut(title)
        tempList = cleanStopWord(tempList)
        for word in tempList:
            if word in WordsFrenquencyDict.keys():  # 使用in判断word是否存在于tempDict的键中
                WordsFrenquencyDict[word] += 1
            else:
                WordsFrenquencyDict[word] = 1

def cleanWordsFrenquencyDict():
    WordsFrenquencyDict = []

def cleanStopWord(wordList):
    if not wordList or len(wordList) == 0:
        return None

    newWordList = []
    for word in wordList:
        if word not in StopWord:
            newWordList.append(word)

    return newWordList

def loadStopWord():
    stopWordFile = open("stopWord.txt","r", encoding="utf-8")   #设置文件对象

    if not stopWordFile:
        return False
    if len(StopWord) > 0:
        return False

    words = stopWordFile.readlines()
    for word in words:
        word = word[:-1]
        StopWord.append(word)

    return True


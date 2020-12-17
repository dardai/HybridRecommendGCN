#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import datetime
import time

from changedPredeal import updateCourseDrChanged
from courseDifferStats import differFusion
from changedBigraph import bigraphChangedMain
from resultFusion import fusion
from courseDislikeStats import dislikeFusion

def online_run():
    # 设定秒数
    changedTime = 30
    while True:
        d = time.time()
        d = d - changedTime
        d = datetime.datetime.fromtimestamp(d)
        updateCourseDrChanged(d)
        differFusion(d)
        dislikeFusion()
        bigraphChangedMain()
        data = fusion()
        time.sleep(changedTime)


"""
#设定秒数
changedTime =30
while True:
    d = time.time()
    d = d - changedTime
    d = datetime.datetime.fromtimestamp(d)
    updateCourseDrChanged(d)
    differFusion(d)
    bigraphChangedMain()
    fusion()
    time.sleep(changedTime)
"""

online_run()


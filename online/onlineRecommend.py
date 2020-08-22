#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import datetime

from changedPredeal import updateCourseDrChanged
from courseDifferStats import differFusion
from changedBigraph import bigraphChangedMain
from resultFusion import fusion

d = datetime.datetime(2020, 8, 1, 0, 0, 0)
updateCourseDrChanged(d)
differFusion(d)
bigraphChangedMain()
fusion()
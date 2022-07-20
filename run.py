# -*- coding: utf-8 -*-



from bigraph.coursePredeal import updateCourseDr
from bigraph.courseRecommend import bigraphMain
from DGL.DGLmain import run

# 课程数据的预处理，从多种行为到综合评分的转换
updateCourseDr()

# 二部图扩展算法，得到基础推荐后的结果
bigraphMain()

# GCMC图卷积算法，得到最终的推荐结果
run()


#import os
#os.system('python train.py')



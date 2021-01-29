# -*- coding: utf-8 -*-



from bigraph.coursePredeal import updateCourseDr
from bigraph.courseRecommend import bigraphMain

# run coursePredeal
updateCourseDr()

#run bigraph
bigraphMain()
# from gcn import train
from DGL.DGLmain import run

run()
#import os
#os.system('python train.py')



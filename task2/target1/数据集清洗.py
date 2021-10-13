# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 08:48:59 2021

@author: God bless CHINA
"""

import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
from loader import dataset

pubmed=dataset()
#建立类实例
pubmed.load_data('pubmed')
#读取数据集
Af=np.empty((19717,19717))
#创建用来保存邻接矩阵的数组
for i in range(len(pubmed.graph)):
    for j in range(len(pubmed.graph[i])):
        Af[i][pubmed.graph[i][j]]=1
        #写邻接矩阵
np.savetxt('邻接矩阵.edge',Af,fmt='%d')
#写入邻接矩阵文件
Tf=sp.vstack((pubmed.allx, pubmed.tx))
#将allx和tx合并成为特征矩阵
np.savetxt('特征矩阵.feature',Tf.A)
#写入特征矩阵文件
Bf=np.vstack((pubmed.ally,pubmed.ty))
#将ally和ty合并成为标签矩阵
np.savetxt('标签矩阵.label',Bf)
#写入标签矩阵文件


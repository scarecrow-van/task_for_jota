# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 16:38:35 2021

@author: God bless CHINA
"""

import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys

class dataset:
    x, y, tx, ty, allx, ally, graph, test_index = range(8)
    #初始化类属性
    def load_data(self,dataset_str):
        #GCN的几个数据集都有相似的保存形式，可以共用一种解码函数
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph','test.index']
        #文件名列表
        objects = []  #一个存放解码后的文件的列表
        for i in range(len(names)):
            with open(f"pubmed/ind.{dataset_str}.{names[i]}", 'rb') as f:
                #依次打开各文件
                if names[i]=='test.index':
                    #test.index的存储格式不太一样，使用np.loadtxt读取
                    #np.loadtxt逐行读取文件并保存为数组，期望每行的数据有相同格式
                    data=np.loadtxt(f)
                else:
                    #别的文件都采用pkl.load读取，latin1是ASCII码的向上兼容编码
                    #pickle.load从文件中读取一个字符串，并将它重构为原来的python对象
                    data = pkl.load(f, encoding='latin1')
                objects.append(data)
                #对于每个读取的data，将其加入objects列表
        self.x, self.y, self.tx, self.ty, self.allx, self.ally, self.graph, self.test_index = objects
        #依次将data赋给对应名字的变量
        #将data赋给对应名字的类属性
        
    
    
    
    
                #if(names[i]!='graph'):
                    # for j in range(data.shape[0]):
                    # print('********',names[i],j,data[j].shape,'**********')
                    # print(data[j])
                # else:
                #     print(f)
                #     print(data)
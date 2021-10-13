# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 15:08:21 2021

@author: God bless CHINA
"""

from sklearn import svm
import numpy as np
import loader

if __name__ == '__main__':
    train_images = loader.load_train_images()
    train_labels = loader.load_train_labels()
    test_images = loader.load_test_images()
    test_labels = loader.load_test_labels()
    m = 1000  # 创建一个读入数据的数组，部分选取图片进行训练。m为选取数目。
    trainingMat = np.zeros((m, 784))
    #初始化存放部分选取图片的数组并拉直，即(m,n*n)的数组。置为0。
    part_train_labels=train_labels[0:m]  #直接截取出存放这部分图片标签的数组。
    for i in range(m):
        for j in range(28):
            for k in range(28):
                trainingMat[i, 28*j+k] = train_images[i][j][k]
    #将前面m张图片赋给存放部分图片的数组。

    mTest = 100 #选取mTest个图片进行测试。
    vectorUnderTest = np.zeros((mTest,784)) 
    #初始化存放部分选取图片的数组并拉直，即(m,n*n)的数组。置为0。
    for i in range(mTest):
        for j in range(28):
            for k in range(28):
                vectorUnderTest[i,28*j+k] = test_images[i][j][k]  
    #将前面mTest张图片赋给存放部分图片的数组
    
    # 获取一个支持向量机模型
    predictor = svm.SVC(gamma='scale', C=1.0, decision_function_shape='ovr', kernel='rbf')
    # 把数据丢进去
    predictor.fit(trainingMat, part_train_labels) #注意这里需要的数组格式正好是二维
    # 预测结果
    result = predictor.predict(vectorUnderTest) #注意这里需要的数组格式正好是二维
    # 准确率估计
    accurancy = np.sum(np.equal(result, test_labels[:mTest])) / mTest
    print(accurancy)

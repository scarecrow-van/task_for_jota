# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 11:33:17 2021

@author: God bless CHINA
"""

import torch
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
#from torch.autograd import Variable

batch_size=64
#下载测试集
test_dataset = datasets.MNIST(root='../num/',train=False,transform=transforms.ToTensor(),download=True)
#../是上级相对路径
test_loader = DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True)

###############可参考：读取模型；开始测试#################
net = torch.load('alexnet_MNIST.pkl')
with torch.no_grad():
    #在接下来的代码中，所有Tensor的requires_grad都会被设置为False
      correct = 0
      total = 0
      for data_test in test_loader:
        images, labels = data_test
        #images, labels = Variable(images), Variable(labels) #pytorch 1.8 loss包括了这功能
        output_test = net(images)
        _, predicted = torch.max(output_test, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
      print("correct1: ", correct)
      print("Test acc: {0}".format(correct.item()/len(test_dataset)))
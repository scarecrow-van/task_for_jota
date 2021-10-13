# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 09:44:14 2021

@author: God bless CHINA
"""

import torch
from torch import nn
from torch import optim
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
#from torch.autograd import Variable
from lenet import Lenet



###################可参考的第一步：打包数据以及超参数#####################
batch_size=64
LR=0.01 #设置学习率
epoch = 1 #分两批计算

# 下载训练集
train_dataset = datasets.MNIST(root='../num/',train=True,transform=transforms.ToTensor(),download=True)
#../是上级相对路径
train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)

################可参考的第二步：配置网络的优化器和loss#######################
net=Lenet(1,10) #图片大小28*28，数据种类为10（1~9）
criterion=nn.CrossEntropyLoss() #交叉熵损失函数
optimizer=optim.Adam(params=net.parameters(),lr=LR) #Adam优化器。（parameters是存储网络的参数）


if __name__ == '__main__':
    ################可参考的第三步：模型训练##############
  for epoch in range(epoch): #对每一批数据
    sum_loss = 0.0
    for i, data in enumerate(train_loader):
      inputs, labels = data #将data里的数据提取出来
      #inputs, labels = Variable(inputs), Variable(labels) #都进行求梯度
      optimizer.zero_grad() #将梯度归零
      outputs = net(inputs) #将数据传入网络进行前向运算
      loss = criterion(outputs, labels) #得到损失函数（在1.8的pytorch损失函数这一步有了求导功能）
      loss.backward() #反向传播
      optimizer.step() #通过梯度做一步参数更新
      
      
      ###############可参考的第四步：打印损失############
      # print(loss)
      sum_loss += loss.item() #每一步的损失
      if i % 100 == 99:
        print('[%d,%d] loss:%.03f' %
           (epoch + 1, i + 1, sum_loss / 100)) #打印训练时每100张的损失
        sum_loss = 0.0
  
    ###########可参考的第五步：保存网络##############
  torch.save(net, 'lenet_MNIST.pkl')  #保存lenet网络的数据
  
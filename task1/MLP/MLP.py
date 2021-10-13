# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 11:05:10 2021

@author: God bless CHINA
"""

import torch
from  torch.autograd import *
from  torch import nn,optim
from  torch.utils.data import DataLoader
from  torchvision import datasets,transforms
 

class MLP(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1),nn.BatchNorm1d(n_hidden_1),nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2),nn.BatchNorm1d(n_hidden_2),nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))
        #三层MLP，使用了归一化处理避免梯度消失并且加快训练速度
 
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
    #前向传播
 
batch_size=64
learning_rate=1e-2
num_epoches=20
 
data_tf=transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5],[0.5])])
train_dataset=datasets.MNIST(root='../num/',train=True,transform=data_tf,download=True)
test_dataset=datasets.MNIST(root="../num/",train=False,transform=data_tf)
train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=batch_size,shuffle=False)
#数据处理

model=MLP(28*28,300,100,10)
criterion=nn.CrossEntropyLoss()
#交叉熵损失
optimizer=optim.SGD(model.parameters(),lr=learning_rate)
#SGD优化器
 
for epoch in range(num_epoches):
    loss_sum, cort_num_sum,acc = 0.0, 0,0
    for data in train_loader:
        img,label=data
        img=img.view(img.size(0),-1)
        inputs = Variable(img)
        target = Variable(label)
        output =model(inputs)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += loss.data
        _, pred = output.data.max(1)
        num_correct = pred.eq(target).sum()
        cort_num_sum += num_correct
    acc=cort_num_sum.float()/len(train_dataset)
    print( "After %d epoch , training loss is %.2f , correct_number is %d  accuracy is %.6f. "%(epoch,loss_sum,cort_num_sum,acc))
 
 
model.eval()
eval_loss=0
eval_acc=0
for data in test_loader:
    img ,label =data
    img=img.view(img.size(0),-1)
    img = Variable(img, volatile=True)
    label = Variable(label, volatile=True)
    out=model(img)
    loss=criterion(out,label)
    eval_loss+=loss.data*label.size(0)
    _,pred=out.data.max(1)
    num_correct=pred.eq(label).sum()
    eval_acc+=num_correct.data
print('Test loss: {:.6f},ACC: {:.6f}'.format(eval_loss.float()/(len(test_dataset)),eval_acc.float()/(len(test_dataset))))
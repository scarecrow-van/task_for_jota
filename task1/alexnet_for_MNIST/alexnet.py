# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 15:42:13 2021

@author: God bless CHINA
"""
#注意：策略是适应MNIST数据集大小调整参数，与论文的alexnet不同
import torch.nn as nn
class AlexNet(nn.Module):
    def __init__(self,in_dim,n_class):
        super(AlexNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, 32, 3, 1),
            #in_channels, out_channels, kqernel_size, stride, padding=0
            # 28*28传入，26*26传出
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # kernel_size, stride
            ##减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
            #26*26传入，13*13传出
            nn.Conv2d(32, 64, 3, 1, 1),
            #13*13传入，13*13传出
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            ##连续3个卷积层，且使用更小的卷积窗口。除了最后的卷积层外，进一步增大了输出通道数。
            #13*13传入，6*6传出
            nn.Conv2d(64, 128, 3, 1, 1),
            #6*6传入，6*6传出
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, 1, 1),
            #6*6传入，6*6传出
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            #6*6传入，6*6传出
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
            #6*6传入，3*3传出
            
        )
        self.fc = nn.Sequential(
            nn.Linear(256*3*3, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, n_class),
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output

#########特别标注：原始论文的参数#####################
'''
        self.conv1 = nn.Conv2d(3, 96,11,4,0)  
        ##调整规律：3*32的输出调整为1*32的输出
        ##227相对于11的卷积调整为28相对于3的卷积
        ##步长由4调整为1
        #227*227传入，55*55传出
        self.pool1 = nn.MaxPool2d(3, 2)
        #55*55传入，27*27传出
        self.conv2 = nn.Conv2d(96, 256, 5, 1, 2)
        #27*27传入，27*27传出
        self.pool2 = nn.MaxPool2d(3, 2)
        #27*27传入，13*13传出
        self.Conv3 = nn.Conv2d(256, 384, 3, 1, 1)
        #13*13传入，13*13传出
        self.Conv4 = nn.Conv2d(384,384, 3, 1, 1)
        #13*13传入，13*13传出
        self.Conv5 = nn.Conv2d(384,256, 3, 1, 1)
        #13*13传入，13*13传出
        self.pool3 = nn.MaxPool2d(3, 2)
        #13*13传入，6*6传出
        self.drop = nn.Dropout(0.5)
        #舍弃一半的数据，但与尺寸无关
        self.fc1 = nn.Linear(9216, 4096)
        #9216=256*6*6。
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 100)
'''

import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
from torch.nn.parameter import Parameter
import torch
import math


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, out, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, out)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        #relu激活函数
        x = F.dropout(x, self.dropout, training = self.training)
        #dropout层
        x = self.gc2(x, adj)
        return x

# class GCN(nn.Module)定义了一个图卷积神经网络，其有两个卷积层：

# 卷积层1：输入的特征为nfeat，维度是2708，输出的特征为nhid，维度是16；
# 卷积层2：输入的特征为nhid，维度是16，输出的特征为nclass，维度是7（即类别的结果）
# forward是向前传播函数，最终得到网络向前传播的方式为：relu–>fropout–>gc2–>softmax

class Attention(nn.Module):
    #注意力机制
    def __init__(self, in_size, hidden_size=10):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            #一个全连接层，输入in_size，输出hidden_size
            nn.Tanh(),
            #tanh激活函数
            nn.Linear(hidden_size, 1, bias=False)
            #全连接层，输入hidden_size，输出1
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta
    #前向传播

class SFGCN(nn.Module):
    def __init__(self, nfeat, nclass, nhid1, nhid2, n, dropout):
        super(SFGCN, self).__init__()

        self.SGCN1 = GCN(nfeat, nhid1, nhid2, dropout)
        self.SGCN2 = GCN(nfeat, nhid1, nhid2, dropout)
        self.CGCN = GCN(nfeat, nhid1, nhid2, dropout)
        #使用了三层GCN

        self.dropout = dropout
        #dropout层
        self.a = nn.Parameter(torch.zeros(size=(nhid2, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        #参数保存
        self.attention = Attention(nhid2)
        #注意力机制
        self.tanh = nn.Tanh()
        #激活层
        self.MLP = nn.Sequential(
            nn.Linear(nhid2, nclass),
            nn.LogSoftmax(dim=1)
        )
        #全连接层

    def forward(self, x, sadj, fadj):
        emb1 = self.SGCN1(x, sadj) # Special_GCN out1 -- sadj structure graph
        com1 = self.CGCN(x, sadj)  # Common_GCN out1 -- sadj structure graph
        com2 = self.CGCN(x, fadj)  # Common_GCN out2 -- fadj feature graph
        emb2 = self.SGCN2(x, fadj) # Special_GCN out2 -- fadj feature graph
        Xcom = (com1 + com2) / 2
        ##attention
        emb = torch.stack([emb1, emb2, Xcom], dim=1)
        emb, att = self.attention(emb)
        output = self.MLP(emb)
        return output, att, emb1, com1, com2, emb2, emb
    #前向传播

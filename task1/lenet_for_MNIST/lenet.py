from torch import nn


#####################可参考：网络如何搭建（框架下增添函数、参数改动等）#################
class Lenet(nn.Module):
    def __init__(self,in_dim,n_classes): #in_dim指输入数据体深度，n_classes指分类数量
        super().__init__()  #这里用了父类的初始化方法
        self.conv=nn.Sequential(
            nn.Conv2d(in_channels=in_dim,out_channels=6,kernel_size=3,stride=1,padding=1),
            #输入深度为in_dim，输出为6，卷积核尺寸为3，滑动步长为1，四边扩充
            #28*28的数据传入，28*28传出
            nn.ReLU(True),
            #ReLU激活函数
            nn.MaxPool2d(kernel_size=2,stride=2),
            #池尺寸为2，步长为2，28*28缩小成14*14
            nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5,stride=1,padding=0),
            #输入深度为之前的输出，输出为16，卷积核尺寸为5，滑动步长为1，四边不扩充
            #14*14传入，10*10传出
            nn.ReLU(True),
            nn.MaxPool2d(2,2))
            #10*10传入，5*5传出。
        #7层结构中的前四层，卷池卷池。卷后面要加激活函数。
        self.full_conect=nn.Sequential(
            nn.Linear(400,120), #5*5的*16的深度=400的传入，120传出
            nn.ReLU(1),
            nn.Linear(120,84), #120传入，84传出
            nn.ReLU(1),
            nn.Linear(84,n_classes)) #84传入，分类类别数目传出
        
    #################下面是前向传播内容#################
    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0),400) #view函数使out形状改变，out.size(0)返回了out的第一个维度的值。
        out = self.full_conect(out)
        return out
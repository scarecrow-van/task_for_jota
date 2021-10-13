import numpy as np
np.set_printoptions(threshold=np.inf)
import loader


#注释习惯：跨行注释归属于上方板块

class Kmeans:
    def __init__(self,k,maxiter,data,labels):
        self.k = k  #k是簇的数目
        self.maxiter=maxiter  #最大迭代次数
        self.data=data  
        #输入的数据，是一个(m,n*n)的数组，m代表有m个测试图，n*n是特征。
        #（例如由图片28*28二维矩阵转化为这个784一维矩阵）
        self.labels=labels  #标签文件，也是一个对应位置的一维矩阵。
        self.distances=np.zeros((self.data.shape[0],self.k))
        #data.shape[0]代表测试图数目（前文注释的m）。
        #data.shape[1]代表测试图特征（前文注释的n*n）。
        #创建个空的(m,k)数组，用来保存距离。
        self.centre=np.zeros((self.k,self.data.shape[1]))
        #创建个空的(k,n*n)数组，用来保存中心点。
        
    def get_distances(self):  
        #计算每幅图到每个中心图距离的函数，算得的距离保存为一个(m,k)数组。
        #这个数组在m上的索引顺序与data相同，即还是按照原来的顺序对应图片。
        for i in range(self.data.shape[0]):  #对每幅图进行计算。
            distance_i=((np.tile(self.data[i],(self.k,1))-self.centre)**2).sum(axis=1) ** 0.5
            #得到一个长度k的一维数组（几个关键函数好好看看）。
            self.distances[i]=distance_i  
            #将得到的一维数组放在distances的对应位置，k轴从0到k-1。
            
    def get_centre(self): #初始化中心点，并初始化分类数组。
        self.classifications=np.random.randint(0,self.k,(self.data.shape[0]))
        #创建一个(m,)的分类数组，里面填充[0,k)的随机整数，代表每个图的初始化聚类。
        for i in range(self.k):
            self.classifications[i]=i
            #防止出现空的聚类使后面中心点计算报错。
    
    
    def classify(self):  #分类的函数。
        new_classifications=np.argmin(self.distances,axis=1)
        #计算【距离数组里每一行的最小值的索引】所组成的一维数组。
        if any(self.classifications-new_classifications):
            self.classifications=new_classifications
            #如果得到的一维数组与之前的分类数组不完全相同，则该数组作为新分类数组。
            return 1
            #返回值控制外部循环。
        else:
            return 0
        #如果完全相同就跳过，不用再替换了。返回值控制外部循环。
        
    def update_centre(self):  #更新中心点的函数。
        for i in range(self.k):
            self.centre[i] = np.mean(self.data[self.classifications==i], axis=0 )
            #每个聚类的中心点是【所有标签为该聚类的点的中心】，即数据轴的平均值。
    def work(self):  #kmeans的计算函数。
        self.get_centre() #先初始化中心点。
        for i in range(self.maxiter):  #控制次数。
            self.update_centre()  #更新中心点。
            self.get_distances()  #求距离。
            if(not self.classify()):  #根据距离分类。
                break
            #如果分类不变化则停止for循环。
    def check(self):  #检查函数（未完成）
        t=1
        for i in range(self.data.shape[0]):
            print('label:%d   predict:%d   num:%d'%(self.labels[i],self.classifications[i],t))
            t+=1
            

if __name__ == '__main__':
    train_images = loader.load_train_images()
    train_labels = loader.load_train_labels()
    test_images = loader.load_test_images()
    test_labels = loader.load_test_labels()

    m = 1500  # 创建一个读入数据的数组，部分选取图片进行训练。m为选取数目。
    n_clusters=30 #聚类的数目，即k值
    trainingMat = np.zeros((m, 784))  
    #初始化存放部分选取图片的数组并拉直，即(m,n*n)的数组。置为0。
    part_train_labels=train_labels[0:m]  #直接截取出存放这部分图片标签的数组。
    for i in range(m):
        for j in range(28):
            for k in range(28):
                trainingMat[i, 28*j+k] = train_images[i][j][k]
    #将前面m张图片赋给存放部分图片的数组。
    mTest = 100
 
    for i in range(mTest):
        classNumStr = test_labels[i]
        vectorUnderTest = np.zeros(784)
        for j in range(28):
            for k in range(28):
                vectorUnderTest[28*j+k] = test_images[i][j][k]  #第i幅测试图  
    a=Kmeans(n_clusters,300,trainingMat,part_train_labels)
    a.work()
    #a.check()
    
    ##################下面是检测预测结果的内容#################
    label_num = np.zeros((n_clusters, 10))
    for i in range(a.classifications.shape[0]):
        pred = int(a.classifications[i])
        truth = int(part_train_labels[i])
        label_num[pred][truth] += 1
    ## 查看KNN label---> number label的对应关系
    label2num = label_num.argmax(axis=1)       
    set( label2num ) ## 看下分类是否覆盖10个数字
    train_preds = np.zeros(part_train_labels.shape)
    for i in range(train_preds.shape[0]):
        train_preds[i] = label2num[a.classifications[i]]
    
    print("训练数据上的精度：{}".format(np.sum(train_preds==part_train_labels) / part_train_labels.shape[0]))
    
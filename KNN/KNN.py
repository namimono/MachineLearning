import KNND
from  numpy import *
import collections


def classify(inX,dataSet,labels,k):
    numOfData=dataSet.shape[0]#获取训练集的行数，即获取训练集中数据个数
    diffxd=tile(inX,(numOfData,1))-dataSet#复制输入数据分别与训练集中的数据相减
    distances=((diffxd**2).sum(axis=1))**0.5#求出所有距离的集合
    sortDistances=distances.argsort()#从小到大排序，然后按索引值输出
    #classCount={}
    voteLabels=[]
    for i in range(k):
        voteLabels.append(labels[sortDistances[i]])
        
        #classCount[voteLabels]=classCount.get(voteLabels,0)+1
    
    classCount2=collections.Counter(voteLabels)
    
    res=sorted(classCount2.items(),key=lambda item:item[1],reverse=True)
    return res[0][0]
    
    
    
    
    
    
    
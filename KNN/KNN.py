import KNND
from  numpy import *
import collections


def classify(inX,dataSet,labels,k):
    numOfData=dataSet.shape[0]#��ȡѵ����������������ȡѵ���������ݸ���
    diffxd=tile(inX,(numOfData,1))-dataSet#�����������ݷֱ���ѵ�����е��������
    distances=((diffxd**2).sum(axis=1))**0.5#������о���ļ���
    sortDistances=distances.argsort()#��С��������Ȼ������ֵ���
    #classCount={}
    voteLabels=[]
    for i in range(k):
        voteLabels.append(labels[sortDistances[i]])
        
        #classCount[voteLabels]=classCount.get(voteLabels,0)+1
    
    classCount2=collections.Counter(voteLabels)
    
    res=sorted(classCount2.items(),key=lambda item:item[1],reverse=True)
    return res[0][0]
    
    
    
    
    
    
    
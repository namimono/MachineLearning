import KNN
from numpy import *
from pip._vendor.distlib.compat import raw_input


def classifyPerson():
    dataSet,labels=parseFile("data.txt")
    dataSet=autoNorm(dataSet)
    a=float(raw_input("enter a"))
    b=float(raw_input("enter b"))
    c=float(raw_input("enter c"))
    inX=array([a,b,c])
    res=KNN.classify(inX, dataSet, labels, 3)
    
    print(res)
def parseFile(filename):
    f=open(filename)
    fileLine=f.readlines()
    numOfLine=len(fileLine)
    rematrix=zeros((numOfLine,3))
    classLabels=[]
    index=0
    for line in fileLine:
        line=line.strip()
        line=line.split(',')
        
        rematrix[index]=line[0:3]
        #classLabels.append(ord(line[-1][0]))
        classLabels.append(line[-1])
        index+=1
    
    return rematrix,classLabels

def autoNorm(dataSet):
    minval=dataSet.min(axis=0)
    maxval=dataSet.max(axis=0)
    
    line=dataSet.shape[0]
    newDataSet=(dataSet-tile(minval,(line,1)))/tile((maxval-minval),(line,1))
    return newDataSet

    #newValue=(oldvalue-min)/(max-min)
    
    
    
    
    
    
    
    
    
    
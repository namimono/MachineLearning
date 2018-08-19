import numpy as ny
import matplotlib.pyplot as plt

def loadDataSet(filename):
    dataset=[]
    labels=[]
    f=open(filename)
    file_lines=f.readlines()
    for x in file_lines:
        x=x.strip().split()
        dataset.append([1.0,float(x[0]),float(x[1])])
        labels.append(int(x[-1]))
    return dataset,labels


def sigmoid(x):
    return 1.0/(1+ny.exp(-x)) 

def gradAscent(dataset,labels):  
    data_mat=ny.mat(dataset)
    label_mat=ny.mat(labels).transpose()
    l_row,l_col=ny.shape(data_mat)
    alpha=0.001
    max_cycles=500
    weight=ny.ones((l_col,1))
    for k in range(max_cycles):
        h=sigmoid(data_mat*weight)
        err=(label_mat-h)
        weight=weight+alpha*data_mat.transpose()*err
    return weight
        
        
def stocGradAscent(dataset,labels,numIter=150):
    dataset=ny.array(dataset)
       
    m,n=ny.shape(dataset)
    
    weight=ny.ones(n)
    for j in range(numIter):
        data_index=list(range(m))
        for i in range(m):
            alpha=4/(1.0+j+i)+0.01
            randIndex=int(ny.random.uniform(0,len(data_index)))
            h=sigmoid(ny.sum(dataset[randIndex]*weight))
            err=labels[randIndex]-h 
            weight=weight+alpha*dataset[randIndex]*err
            del(data_index[randIndex])
    return weight    
        
def plotBestFit(weight,list_data,list_labels):
    
    arr_data=ny.array(list_data)
    len_data=len(arr_data)
    x1=[];x2=[]
    y1=[];y2=[]
    for i in range(len_data):
        if list_labels[i]==1:
            x1.append(arr_data[i,1])
            y1.append(arr_data[i,2])
        else:
            x2.append(arr_data[i,1])
            y2.append(arr_data[i,2])
            
    fig=plt.figure()      
    ax=fig.add_subplot(111)
    ax.scatter(x1,y1,s=20,c='red',marker='s')
    ax.scatter(x2,y2,s=10,c='blue')
    plt.title("dataset")
    plt.xlabel('x1')
    plt.ylabel('x2')
    x=ny.arange(-3.0,3.0,0.1)
    x2=(-weight[0]-weight[1]*x)/weight[2]
    ax.plot(x,x2)
    plt.show()
    

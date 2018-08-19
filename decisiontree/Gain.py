from math import log
import collections
def informationEntropy(dataset):
    num_of_line=len(dataset)
    col_category=[x[-1] for x in dataset]
    labels=collections.Counter(col_category)
    '''labels={}
    for x in dataset:
        current_labels=x[-1]
        labels[current_labels]=labels.get(current_labels, 0)+1'''
    entropy=0.0
    for key in labels:
        prob=float(labels[key]/num_of_line)
        
        entropy+=-prob*log(prob,2)
    return entropy


def splitDataSet(dataset,feature,value):
    
    new_data_set=[]
    for x in dataset:
        if x[feature]==value:
            temp=[]
            temp.append(x[feature])
            temp.append(x[-1])
            new_data_set.append(temp)
    return new_data_set

def informationGain(dataset,feature):
    
    feature_column=[ex[feature] for ex in dataset]#抽取特征列
    line_of_data=len(dataset)                        #获得所有数据的个数
    base_entropy=informationEntropy(dataset)        #计算整体熵
    feature_dict=collections.Counter(feature_column)  #统计特征列

        #计算条件熵
    con_entropy=0.0
    for x in feature_dict:
        chil_data=splitDataSet(dataset, feature, x)
        con_entropy+=(feature_dict[x]/line_of_data)*informationEntropy(chil_data)
    return base_entropy-con_entropy

def chooseBestFeature(dataset):
    num_of_feature=len(dataset[0])-1
    
    best_info_gain=-1;best_feature=None
    for y in range(num_of_feature):
        info_gain=informationGain(dataset, y)
        if(info_gain>best_info_gain):
            best_info_gain=info_gain
            best_feature=y
    return best_feature  


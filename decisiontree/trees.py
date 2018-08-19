import Gain as Ga
import collections
def creatTree(dataset):
    class_list=[x[-1] for x in dataset]
    if class_list.count(class_list[0])==len(class_list):
        return class_list[0]
def predict(data,node):
    
    if node.isnode==0:
        return node.result
    for x in node:
        if data[node.best_feature]==x.getNode()[0][node.best_feature]:
            return predict(data,x)

def splitNodeData(dataset,feature,value):
    new_dataset=[]
    for x in dataset:
        if x[feature]==value:
            new_dataset.append(x)
        
    return new_dataset
class Node:
    def __init__(self,dataset=[]):
        
        self.__dataset=dataset
        self.__child=[]
        self.result=None   
        self.isnode=1
        self.best_feature=None
    def __getitem__(self,position):
        return self.__child[position]
    def __getattr__(self,name):
        return 
    def getNode(self):
        return self.__dataset
    def getNodeChild(self):
        return self.__child
'''    def __iter__(self):
        return iter(self.__child)'''       #委托迭代
    
class DecisionTree:
    def __init__(self):
        pass
    def buildTree(self,node):
        class_list=[x[-1] for x in node.getNode()]
        #递归结束 判断
        if class_list.count(class_list[0])==len(class_list):
            node.isnode=0
            node.result=class_list[0]
            return 
        #child=[node1,node2,node3]
        best_feat=Ga.chooseBestFeature(node.getNode())
        node.best_feature=best_feat
        #统计特征列
        
        feat_column=[x[best_feat] for x in node.getNode()]
        feature_dict=collections.Counter(feat_column)
        
        for x in feature_dict:
            
            child_data=splitNodeData(node.getNode(), best_feat, x)
            
            child=Node(child_data)
            node.getNodeChild().append(child)
            
                
        for x in node.getNodeChild():
            
            self.buildTree(x)

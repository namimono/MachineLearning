import re
import collections
import numpy as ny
from array import array
def parseFile(filename):
    f=open(filename)
    fileline=f.readlines()
    
    
    dataset=[];labels=[]
    regEx_1=re.compile(r'\W')
    for x in fileline:
        x=x.strip()
        
        x=regEx_1.split(x)
        x=[a for a in x if len(a)>0]
        labels.append(x.pop())
        
        dataset.append(x)
    return dataset,labels
    
def creatVocabList(dataset):
    vocab_list=set([])
    for x in dataset:
        vocab_list=vocab_list|set(x)
    return list(vocab_list)

def setWords2Bag(vocab_list,dataset):
    
    word_bag=[]
    for x in dataset:
        vec_word=[0]*len(vocab_list)
        for y in x:                      
            if y in vocab_list:
                vec_word[vocab_list.index(y)]+=1
        word_bag.append(vec_word)
    return word_bag
def setWords2Bag2(vocab_list,dataset):
    
    word_bag=[0]*len(vocab_list)
    for x in dataset:                     
        if x in vocab_list:
            word_bag[vocab_list.index(x)]+=1
    return word_bag    

def trainNB(dataset,labels,vocab_list):
    index=[]
    word_bag=setWords2Bag(vocab_list, dataset)
    dict_labels=collections.Counter(labels)
    pDoc=[]
    for x in dict_labels:
        index.append(x)
        pDoc.append(dict_labels[x]/len(labels))
    pNum=[]
    pAll=[]
    #P=ny.zeros((len(dict_labels),len(vocab_list)))

    for key in dict_labels:
        
        pn=ny.ones(len(vocab_list))
        pa=2.0;pa2=[]
        for x in range(len(labels)):
            if labels[x]==key:
                pn+=word_bag[x]
                pa+=sum(word_bag[x])
        pa2.append(pa)        
        pNum.append(pn)
        pAll.append(pa2)
    pVect=ny.log(ny.array(pNum)/ny.array(pAll))
    return pVect,pDoc,index
def predict(pVect,pDoc,index,doc):
    p=[]
    for x in range(len(pVect)):
        
        p.append((sum(doc*pVect[x])+ny.log(pDoc[x])))
    p=ny.array(p)
    return index[p.argsort()[-1]]
    
    
    
            
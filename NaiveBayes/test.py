import bayes as by
import numpy as ny


test_doc1=['love','my','dalmation']
test_doc2=['love','dog','my','dog']


dataset,labels=by.parseFile('data')
vocab_list=by.creatVocabList(dataset)

test_doc1=ny.array(by.setWords2Bag2(vocab_list, test_doc2))

pVec,pDoc,index=by.trainNB(dataset, labels,vocab_list)
result=by.predict(pVec,pDoc,index,test_doc1)
print(result)
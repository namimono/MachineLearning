import KNN
import KNND
dataSet,labels=KNND.parseFile("data.txt")
inX=[1,2,9]
print(KNN.classify(inX, dataSet, labels, 9))
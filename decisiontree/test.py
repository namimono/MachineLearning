import trees as tr


a=[[1,1,1,'yes'],
   [1,0,1,'yes'],
   [0,1,0,'yes'],
   [0,1,1,'no'],
   [0,0,1,'no']]
aa=[[1,1,'yes'],
   [1,1,'yes'],
   [1,1,'maybe'],]
aaa=[[1,1,'yes'],
   [1,1,'yes'],
   [1,0,'yes'],
   [1,1,'maybe'],]
b=[['yes','yes','maybe'],
   ['yes','yes','maybe'],
   ['no','no','yes'],
   ['no','yes','no'],
   ['no','yes','no']]

'''tree=tr.DecisionTree()
node0=tr.Node(a)



child1=tr.splitNode(node0.getNode(), 0, 1)
print(child1)
child3=tr.Node(child1)
node0.getNodeChild().append(child3)


child2=tr.splitNode(node0.getNode(), 0, 0)
child4=tr.Node(child2)
node0.getNodeChild().append(child4)
    

'''

tree=tr.DecisionTree()
node0=tr.Node(a)
tree.buildTree(node0)
print(node0.getNode())
'''
test=['no','yes','']

res=tr.predict(test, node0)
print(res)
'''


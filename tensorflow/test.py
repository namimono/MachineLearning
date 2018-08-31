# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as ny
from numpy.random import RandomState
'''a=tf.constant([1,2])
b=tf.constant([2,3])
result=tf.add(a,b,name='add')
result2=tf.add(a,b,name='add')
'''
#声明
from tensorflow.python.training.training_util import global_step
num_input_node=2
num_output_node=1

layer1_node=3
layer2_node=1

#定义学习率，学习率衰减速度，正则系数，训练调整参数的次数以及平滑衰减率
LEARNING_RATE_BASE=0.1
LEARNING_RATE_DECAY=0.96
DECAY_STEP=100
REGULARIZATION_RATE=0.001
global_step=tf.Variable(0)

x=tf.placeholder(tf.float32,shape=(None,num_input_node),name='input')
labels=tf.placeholder(tf.float32,shape=(None,num_output_node),name='labels')
w1=tf.Variable(tf.random_normal([num_input_node,layer1_node],stddev=1))
w2=tf.Variable(tf.random_normal([layer1_node,layer2_node],stddev=1))
#激活函数
a=tf.nn.relu(tf.matmul(x,w1))
y=tf.nn.relu(tf.matmul(a,w2))
#计算交叉熵并求平均值 定义损失函数
cross_entropy=tf.reduce_mean(labels*tf.log(tf.clip_by_value(y, 1e-10 , 1.0)))
regularizer=tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
regularization=regularizer(w1)+regularizer(w2)
loss=cross_entropy+regularization 
# loss2=tf.reduce_mean(tf.square(y - labels))
#生成动态学习率
learning_rate=tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, DECAY_STEP, LEARNING_RATE_DECAY)
train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
# train_step2=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss2,global_step=global_step)
#生成随机数据集和labels集
rdm=RandomState(1)
datasize=200
X=rdm.rand(datasize,2)
LAB=ny.array([[1],[1],[0],[0],[0],[1],[1],[1],[1],[0],[0],[0],[0],[1],[0],[1],[1],[0],[1],[1]])
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    x_feed={x:X[:20]}
    labels_feed={labels:LAB[:20]}
    value_feed={x:X[:20],labels:LAB[:20]}
    print(type(X))
    for i in range(20):
        
        
        b=sess.run(train_step,feed_dict=value_feed)
        print(sess.run(w1))
        '''print(sess.run(w2))
        
        print(sess.run(learning_rate))'''
    
    '''for i in range(0,20,4):
        print(sess.run(w2))
        x_feed={x:X[i:i+4]}
        labels_feed={labels:LAB[i:i+4]}
        value_feed={x:X[i:i+4],labels:LAB[i:i+4]}
        c=sess.run(a,feed_dict=x_feed)
        sess.run(train_step,feed_dict=value_feed)
        
        print(sess.run(w2))
        
        print(sess.run(learning_rate))'''
           
    
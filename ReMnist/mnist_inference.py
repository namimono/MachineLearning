# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

INPUT_NODE=784
OUTPUT_NODE=10
LAYER1_NODE=500

def getWeightVariable(shape,regularizer):
    weights=tf.get_variable('weight',shape,initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer!=None:
        tf.add_to_collection('losses',regularizer(weights))
    return weights

#定义前向传播


def inference(input_tensor,regularizer):

    with tf.variable_scope('layer1'):
        weights=getWeightVariable([INPUT_NODE,LAYER1_NODE], regularizer)
        biases=tf.get_variable('biases',[LAYER1_NODE],initializer=tf.constant_initializer(0.0))
        layer1=tf.nn.relu(tf.matmul(input_tensor,weights)+biases)
        
    with tf.variable_scope('output'):
        weights=getWeightVariable([LAYER1_NODE,OUTPUT_NODE],regularizer)
        biases=tf.get_variable('biases',[OUTPUT_NODE],initializer=tf.constant_initializer(0.0))
        layer2=tf.matmul(layer1,weights)+biases
    return layer2 


        
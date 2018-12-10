# -*- coding: utf-8 -*-
import os

import tensorflow as tf
import tensorboard
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference as infer
BATCH_SIZE=100
LEARNING_RATE_BASE=0.8
LEARNING_RATE_DECAY=0.99
REGULARIZATION_RATE=0.0001
TRAINING_STEP=3000
MOVING_AVERAGE_DECAY=0.99


def train(mnist):
    with tf.name_scope('input'):   
        x=tf.placeholder(tf.float32, [None,infer.INPUT_NODE], 'x_input')
        labels=tf.placeholder(tf.float32, [None,infer.OUTPUT_NODE],'labels')
  
     
    regularizer=tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    avg_y=infer.inference(x, regularizer)
    global_step=tf.Variable(0,trainable=False)
    
    with tf.name_scope('moving_averages'):
        variable_averages=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)   #定义平滑类
        variable_averages_op=variable_averages.apply(tf.trainable_variables())#定义平滑更新变量操作
    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=avg_y)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    with tf.name_scope('train'):
        learning_rate=tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, mnist.train.num_examples/BATCH_SIZE, LEARNING_RATE_DECAY)
        train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
        
        train_op=tf.group(train_step,variable_averages_op)
    
    correct_prediction=tf.equal(tf.argmax(avg_y,1),tf.argmax(labels,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    
    with tf.Session() as sess:
        writer=tf.summary.FileWriter('board',sess.graph)
        tf.initialize_all_variables().run()
        value_feed={x:mnist.validation.images,labels:mnist.validation.labels}
#         test_feed={x:mnist.test.images,labels:mnist.test.labels}
        for i in range(TRAINING_STEP):
            if i%1000==0:
#                 run_options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata_op = tf.RunMetadata()
                validate_acc=sess.run(accuracy,feed_dict=value_feed,options=run_options,run_metadata=run_metadata_op)
                writer.add_run_metadata(run_metadata_op,'step%d'%i)
                print("After %d train steps, validate accuracy using average model is %g"%(i,validate_acc))
                
            xs,ys=mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op,feed_dict={x:xs,labels:ys})
            
        validate_acc=sess.run(accuracy,feed_dict=value_feed)
        print("After %d train steps, validate accuracy using average model is %g"%(i,validate_acc))
    writer.close()   
def main(argv=None):    
    mnist=input_data.read_data_sets("I:\python_project\MINST\mnist_dataset",one_hot=True)
    train(mnist)
if __name__=='__main__':
    tf.app.run()        
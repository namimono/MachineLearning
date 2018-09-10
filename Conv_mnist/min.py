# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

INPUT_NODE=784
OUTPUT_NODE=10

LAYER1_NODE=500

BATCH_SIZE=100

LEARNING_RATE_BASE=0.8
LEARNING_RATE_DECAY=0.99
REGULARIZATION_RATE=0.0001
TRAINING_STEP=2000
MOVING_AVERAGE_DECAY=0.99             #滑动平均衰减
def inference(x,avg_class,weight1,weight2,biases1,biases2):
    if avg_class==None:
        layer1=tf.nn.relu(tf.matmul(x,weight1)+biases1)
        return tf.matmul(layer1,weight2)+biases2
    else:
        layer1=tf.nn.relu(tf.matmul(x,weight1)+avg_class.average(biases1))  #计算滑动平均值
        return tf.matmul(layer1,weight2)+avg_class.average(biases2)

def train(mnist):
    x=tf.placeholder(tf.float32,shape=[None,INPUT_NODE],name="x_input")
    labels=tf.placeholder(tf.float32,shape=[None,OUTPUT_NODE],name="labels")
    
    weight1=tf.Variable(tf.truncated_normal(shape=[INPUT_NODE,LAYER1_NODE],stddev=0.1))
    weight2=tf.Variable(tf.truncated_normal(shape=[LAYER1_NODE,OUTPUT_NODE],stddev=0.1))
    
    biases1=tf.Variable(tf.constant(0.1,shape=[LAYER1_NODE]))
    biases2=tf.Variable(tf.constant(0.1,shape=[OUTPUT_NODE]))
    
    global_step=tf.Variable(0,trainable=False)
    variable_averages=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)   #定义平滑类
    variable_averages_op=variable_averages.apply(tf.trainable_variables())#定义平滑更新变量操作
    
    
    y=inference(x, None, weight1, weight2, biases1, biases2)
    avg_y=inference(x, variable_averages, weight1, weight2, biases1, biases2)
    
    cross_entropy=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=avg_y ,labels=tf.argmax(labels,1)))
    
    #定义正则
    regularizer=tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    regularization=regularizer(weight1)+regularizer(weight2)
    loss=cross_entropy+regularization
    
    learning_rate=tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, mnist.train.num_examples/BATCH_SIZE, LEARNING_RATE_DECAY)
    train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    
    train_op=tf.group(train_step,variable_averages_op)
    
    correct_prediction=tf.equal(tf.argmax(avg_y,1),tf.argmax(labels,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    
    saver=tf.train.Saver()
    with tf.Session() as sess:
        writer = tf.summary.FileWriter('logs/', sess.graph)
        tf.initialize_all_variables().run()
        value_feed={x:mnist.validation.images,labels:mnist.validation.labels}
        test_feed={x:mnist.test.images,labels:mnist.test.labels}
        for i in range(TRAINING_STEP):
            if i%1000==0:
                validate_acc=sess.run(accuracy,feed_dict=value_feed)
                print("After %d train steps, validate accuracy using average model is %g"%(i,validate_acc))
                print(sess.run(weight2))
            xs,ys=mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op,feed_dict={x:xs,labels:ys})
            
        test_acc=sess.run(accuracy,feed_dict=test_feed)
        print("After %d train steps, final accuracy is %g"%(TRAINING_STEP,test_acc))
        saver.save(sess,'save/model.ckpt')
def main(argv=None):    
    mnist=input_data.read_data_sets("I:\python_project\MINST\mnist_dataset",one_hot=True)
    train(mnist)
if __name__=='__main__':
    tf.app.run()

# -- coding: utf-8 --
 
import os
 
import numpy as np
 
import tensorflow as tf  
from tensorflow.examples.tutorials.mnist import input_data  
 
import mnist_inference  
 
#数据batch大小
BATCH_SIZE = 100  
  
#训练参数
LEARNING_RATE_BASE = 0.02  
LEARNING_RATE_DECAY = 0.99   
REGULARIZATION_RATE= 0.0001   
TRAINING_STEPS = 1000
MOVING_AVERAGE_DECAY = 0.99   
 
#模型保存路径及文件名
MODEL_SAVE_PATH = "/model2/"
MODEL_NAME = "model.ckpt"
 
  
def train(mnist): 
#输入层和数据label 
#    x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
    x = tf.placeholder(tf.float32, [None, mnist_inference.IMAGE_SIZE, mnist_inference.IMAGE_SIZE, mnist_inference.NUM_CHANNELS], name='x-input')  
    y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')
 
#前向传播结果y
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)  
    y = mnist_inference.inference(x, 1 ,regularizer)
    global_step = tf.Variable(0, trainable=False)  
 
#滑动平均模型
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)  
    variables_averages_op = variable_averages.apply(tf.trainable_variables())   
 
#计算交叉熵，并加入正则-->损失函数loss
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)   
#    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses')) 
    loss = cross_entropy_mean 
#学习率
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,mnist.train.num_examples/BATCH_SIZE,LEARNING_RATE_DECAY)  
#train_step 梯度下降(学习率，损失函数，全局步数)  
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)  
#运算图控制，用train_op作集合  
    with tf.control_dependencies([train_step, variables_averages_op]):  
        train_op = tf.no_op(name='train')  

#持久化
    saver = tf.train.Saver()
#tf.equal(A, B)是对比这两个矩阵或者向量的相等的元素，如果是相等的那就返回True，反正返回False
#
    correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    
    with tf.Session() as sess:  
        tf.initialize_all_variables().run() 
        x1=mnist.validation.images
        x1=np.reshape(x1, (-1,mnist_inference.IMAGE_SIZE,mnist_inference.IMAGE_SIZE,mnist_inference.NUM_CHANNELS))
        value_feed={x:x1,y_:mnist.validation.labels}
        
        for i in range(TRAINING_STEPS): 
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            reshaped_xs = np.reshape(xs, (BATCH_SIZE, mnist_inference.IMAGE_SIZE, mnist_inference.IMAGE_SIZE, mnist_inference.NUM_CHANNELS))
#             reshaped_xs = tf.reshape(xs, [BATCH_SIZE, mnist_inference.IMAGE_SIZE, mnist_inference.IMAGE_SIZE, mnist_inference.NUM_CHANNELS])
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x:reshaped_xs, y_:ys})
            validate_acc=sess.run(accuracy,feed_dict=value_feed)
            print("After %d train steps, validate accuracy using average model is %g"%(i,validate_acc))   
            print("After %d training step(s), loss on training batch is %g " %(step, loss_value)) 
            if i%200 == 0:
                #将y加入集合，方便加载模型时调用，必须放在这里才能保存最后的y值
                tf.add_to_collection("pred_network", y)
                saver.save(sess, "./model/model.ckpt")

  
def main(argv=None):
    # print(os.getcwd()+"\\MNIST")
    mnist = input_data.read_data_sets("E:\python_project\MachineLearning\RE_CNN\MNIST", one_hot=True)
    train(mnist)  
  
if __name__== '__main__':  
    tf.app.run()
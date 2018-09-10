import tensorflow as tf
INPUT_NODE=784
OUTPUT_NODE=10

IMAGE_SIZE=28
NUM_CHANNELS=3
#conv1
CONV1_DEEP=32
CONV1_SIZE=5

CONV2_DEEP=64
CONV2_SIZE=5


BATCH_SIZE=100

def inference(input_tensor,regularizer):
    with tf.variable_scope('conv1'):
        conv1_filter=tf.get_variable('conv1_filter',shape=[CONV1_SIZE,CONV1_SIZE,NUM_CHANNELS,CONV1_DEEP],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases=tf.get_variable('conv1_biases',shape=[CONV1_DEEP],initializer=tf.constant_initializer(0.0, tf.float32))
        conv1=tf.nn.conv2d(input_tensor,conv1_filter,strides=[1,1,1,1],padding='SAME')
        relu1=tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
    with tf.variable_scope('pool1'):
        pool1=tf.nn.max_pool(relu1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    with tf.variable_scope('conv2'):
        conv2_filter=tf.get_variable('conv2_filter',shape=[CONV2_SIZE,CONV2_SIZE,NUM_CHANNELS,CONV2_DEEP],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases=tf.get_variable('conv2_biases',shape=[CONV2_DEEP],initializer=tf.constant_initializer(0.0, tf.float32))
        conv2=tf.nn.conv2d(pool1,conv2_filter,strides=[1,1,1,1],padding='SAME')
        relu2=tf.nn.relu(tf.nn.bias_add(conv2,conv2_biases))
    with tf.variable_scope('pool2'):
        pool2=tf.nn.max_pool(relu2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        
        
with tf.Session() as sess:
    tf.initialize_all_variables().run()
    
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
with tf.Session() as sess:
  mnist = input_data.read_data_sets("E:\python_project\MachineLearning\RE_CNN\MNIST", one_hot=True)
  #先读取图再读取数据
  new_saver = tf.train.import_meta_graph('model/model.ckpt.meta')
  new_saver.restore(sess, tf.train.latest_checkpoint("./model"))
  # tf.get_collection() 返回一个list. 但是这里只要第一个参数即可
  y = tf.get_collection('pred_network')[0]
  graph = tf.get_default_graph()

  # 因为y中有placeholder，所以sess.run(y)的时候还需要用实际待预测的样本以及相应的参数来填充这些placeholder，而这些需要通过graph的get_operation_by_name方法来获取。
  input_x = graph.get_operation_by_name('x-input').outputs[0]
  input_y = graph.get_operation_by_name('y-input').outputs[0]

  x1=mnist.test.images[0]
  x1 = np.reshape(x1, (1, 28,28,1))
  value_feed={input_x:x1,input_y:mnist.test.labels}

  # correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(input_y, 1))
  # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  # accu=sess.run(accuracy,feed_dict=value_feed)
  y=sess.run(tf.argmax(y, 1),feed_dict={input_x:x1})
  result=sess.run(tf.argmax(input_y,1),feed_dict={input_y:mnist.test.labels[0:10]})
  # print(accu)
  print(y)
  print(result)
  # for i in range(100):
  #   pass
    # 使用y进行预测

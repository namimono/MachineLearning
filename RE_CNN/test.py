import tensorflow as tf
tf.enable_eager_execution()
# ten=tf.Variable([1,2,3,4,556,7,2,112])
ten=[[1,2,3],[2,4,6],[3,4,6],[2,6,9]]
ten3=[[2,4,6],[1,2,3],[3,4,6],[2,6,9]]
a=tf.argmax(ten,1)
c=tf.equal(tf.argmax(ten, 1), tf.argmax(ten3, 1))
print(a)
print(c)
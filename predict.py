import tensorflow as tf
import numpy as np
import data as d

n = 40
c = 5

x = tf.placeholder(tf.float32, [None, n])
W = tf.Variable(tf.zeros([n, c]))
b = tf.Variable(tf.zeros([c]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, c])


saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess,'model.ckpt')


correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
a, b = d.testdata()
print(sess.run(accuracy, feed_dict={x: a, y_: b}))

import tensorflow as tf
import numpy as np
import data as d


def precision_and_recall(actual, predicted):
    for k in set(actual):
        print('Class :', k)
        a = sum([1 if x==k and y==k else 0 for x,y in zip(predicted,actual)])
        b = sum([1 if x==k and y!=k else 0 for x,y in zip(predicted,actual)])
        print('Precision :', a / (a+b))
        c = sum([1 if x==k else 0 for x in actual])
        print('Recall :', a / c)

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
predicted = sess.run(tf.arg_max(y,1), feed_dict= {x:a})
actual = [x.index(1) for x in b]
print('Accuracy')
print(sess.run(accuracy, feed_dict={x: a, y_: b}))
print(precision_and_recall(actual, predicted))

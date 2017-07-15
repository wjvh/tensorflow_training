import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import numpy as np

logdir = '/tmp/test/10'

a = tf.constant(3,tf.float32,name='a')
b = tf.constant(4,tf.float32,name='b')
c = tf.multiply(a,b,name='c')
d = tf.div(a,b,name='d')

sess = tf.Session()
writer = tf.summary.FileWriter(logdir)
writer.add_graph(sess.graph)
#print(sess.run(c))
print(sess.run(d))

# a = [0,0,1]
# c= tf.argmax(a,axis=1)
#
# sess= tf.Session()
# print(sess.run(c))

# a = [0,1,2,1,2,0,2,1]
# a = ['cat','dog','bird','cat','bird','bird']
# num_labels = len(np.unique(a))
# print(num_labels)
# b = np.eye(num_labels)[a]
# print(b)

# from tflearn.datasets import cifar10
# (train_X, train_y), (test_X, test_y) = cifar10.load_data()

# from tflearn.data_utils import shuffle, to_categorical
# X, Y = shuffle(X, Y)
# Y = to_categorical(Y, 10)
# Y_test = to_categorical(Y_test, 10)

# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("mnist", one_hot=True)
# train_X = mnist.train.images
# train_y = mnist.train.labels
# test_X = mnist.test.images
# test_y = mnist.test.labels
# print(train_X[0,:])

# from sklearn import datasets
#
# iris = datasets.load_iris()
# X = iris.data
# y = iris.target
#
# from sklearn.model_selection import train_test_split
# train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25, random_state=42)
#
# print(train_y)

# import matplotlib.pyplot as plt
# plt.scatter(X[:,0],X[:,1],c=y)
# plt.show()

# x = tf.placeholder(tf.float32,shape=[1,2])
# w = tf.constant(
#     [[1,2],
#      [3,4]],tf.float32)
# b = tf.constant([[3.,3.]],tf.float32)
# y = tf.add(tf.matmul(x,w),b)

# a = tf.placeholder(tf.float32)
# b = tf.placeholder(tf.float32)
# d = tf.constant(5)
# c = tf.add(a,b)

# tf.set_random_seed(20)
# a = tf.random_normal([2,2])
# a = tf.truncated_normal([2,2],stddev=0.1)
# b = tf.truncated_normal([2,2],stddev=0.1)
# x = tf.constant(
#     [[1,1]]
# )
# w = tf.constant(
#     [[1,2],
#      [3,4]]
# )
# b = tf.constant(
#     [[2,2]]
# )
# y = tf.add(tf.matmul(x,w),b)
# a = tf.eye(3)
# a = tf.zeros([2,3])
# a = tf.constant(
#     [[1,2],
#      [3,4]],dtype=tf.float32
# )
# d = tf.reduce_sum(a)
# d = tf.reduce_mean(a,1)
# b = tf.constant(
#     [[5,6,],
#     [7,8]]
# )
# c = tf.matmul(a,b)
# a = tf.constant(3)
# b = tf.constant(4)
# c = tf.constant(5)

#d = tf.add(tf.multiply(a,b),c)
# d = (a*b)+c

# sess = tf.Session()
# print(sess.run(y,feed_dict={x:[[1,1]]}))

# print(sess.run(c,feed_dict={a:3,b:4}))
# print(sess.run(c,{a:3,b:4}))
# print(sess.run(b))
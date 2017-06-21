# Tensorflow workshop with Jan Idziak
# -------------------------------------
#
# script based on the:
# Implementation of a simple MLP network with
# one hidden layer.
#
# Linear Regression
# ----------------------------------
#
# This function shows how to use TensorFlow to
# solve linear regression.
# y = Ax + b
# y = Wx
#
# We will use the iris data, specifically:
#  y = Sepal Length
#  x = Pedal Length, Petal Width, Sepal Width
import tensorflow as tf
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)

iris = datasets.load_iris()
data = iris["data"]


features = [[x[0], x[1], x[2]] for x in data]
target = [x[3] for x in data]

train_X, test_X, train_y, test_y = train_test_split(features, target, test_size=0.33, random_state=RANDOM_SEED)

# Symbols
X = tf.placeholder("float", shape=[None, 3])
y = tf.placeholder("float", shape=[None, ])

# Weight initializations
w_1 = tf.Variable(tf.random_normal([3, 1], stddev=0.1))
b = tf.Variable(tf.random_normal([1], stddev=0.1))

# Forward propagation
yhat = tf.matmul(X, w_1) + b
predict = yhat

# Backward propagation
cost = tf.reduce_mean(tf.pow(predict - y, 2))
RMSE = tf.sqrt(cost)
updates = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

# Run SGD
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for epoch in range(80):
        # Train with each example
        for i in range(len(train_X)):
            sess.run(updates, feed_dict={X: train_X[i: i + 1], y: train_y[i: i + 1]})
            train_accuracy = sess.run(RMSE, feed_dict={X: train_X, y: train_y})
            test_accuracy = sess.run(RMSE, feed_dict={X: test_X, y: test_y})

        print("Epoch = %d, train MSE = %.2f, test MSE = %.2f"
              % (epoch + 1, train_accuracy, test_accuracy))
        # print sess.run((y, predict), feed_dict={X: train_X, y: train_y})

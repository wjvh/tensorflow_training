# Module 4: Simple TF Model
# Challenge: MINST Dataset

import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("mnist", one_hot=True)

# Step 1: Model input and output
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
yhat = tf.matmul(x, W) + b

# Step 2: Define Loss
loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))

# Step 3: Optimizer
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

correct_prediction = tf.equal(tf.argmax(yhat, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    # Step 4: Training Loop
    for _ in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train, feed_dict={x: batch_xs, y: batch_ys})

    # Step 5: Prediction
    print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                        y: mnist.test.labels}))
    saver.save(sess, '/tmp/simple_model/model.ckpt')

with tf.Session() as sess:
    saver.restore(sess, '/tmp/simple_model/model.ckpt')
    print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                        y: mnist.test.labels}))

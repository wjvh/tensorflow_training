# Module 4: Simple TF Model

import numpy as np
import tensorflow as tf

# # Step 1: Define Model
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
yhat = W * x + b

# # Step 2: Define Loss
loss = tf.reduce_sum(tf.square(yhat - y)) # sum of the squares

# # Step 3: Optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# # training data
x_train = [1,2,3,4]
y_train = [0,-1,-2,-3]

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# # Step 4: Training Loop
for i in range(1000):
  sess.run(train, {x:x_train, y:y_train})

# Step 5: Prediction
import matplotlib.pyplot as plt
plt.plot(x_train,y_train,'o')
plt.plot(x_train,sess.run(W)*x_train+sess.run(b),'r')
plt.show()


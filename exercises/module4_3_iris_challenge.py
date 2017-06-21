# Iris Flower Dataset - Classification

import tensorflow as tf
import numpy as np
from sklearn import datasets

iris = datasets.load_iris()
data = iris.data
target = iris.target

# Convert the label into one-hot vector
num_labels = len(np.unique(target))
all_Y = np.eye(num_labels)[target]  # One liner trick!

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split

train_X, test_X, train_Y, test_Y = train_test_split(data, all_Y, test_size=0.33, random_state=42)

# Step 1: Define the Model/Graph
X_size = train_X.shape[1]  # Number of input nodes: 4 features and 1 bias
Y_size = train_Y.shape[1]  # Number of outcomes (3 iris flowers)

x = tf.placeholder("float", shape=[None, X_size])
y = tf.placeholder("float", shape=[None, Y_size])

W = tf.Variable(tf.random_normal([X_size, Y_size], stddev=0.1))
b = tf.Variable(tf.random_normal([Y_size], stddev=0.1))
yhat = tf.matmul(x, W) + b

# Step 2: Define Loss
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))

# Step 3: Optimizer
optimizer = tf.train.GradientDescentOptimizer(0.001)
train = optimizer.minimize(loss)

correct_prediction = tf.equal(tf.argmax(yhat, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# Step 4: Training Loop
for epoch in range(150):
    for i in range(len(train_X)):
        sess.run(train, feed_dict={x: train_X[i: i + 1], y: train_Y[i: i + 1]})

print(sess.run(accuracy, feed_dict={x: test_X, y: test_Y}))

# Step 5: Prediction
X = tf.constant([[1., 1., 1., 1.]])
predict = tf.argmax(tf.matmul(X, W), axis=1)
print(sess.run(predict))
sess.close()

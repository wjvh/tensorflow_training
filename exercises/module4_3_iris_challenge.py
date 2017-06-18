# Iris Flower Dataset - Classification

import tensorflow as tf
import numpy as np
from sklearn import datasets


iris = datasets.load_iris()
data = iris.data
target = iris.target

# Prepend the input data with one column of 1 for  bias
N, M = data.shape
all_X = np.ones((N, M + 1))
all_X[:, 1:] = data

# Convert the label into one-hot vector
num_labels = len(np.unique(target))
all_Y = np.eye(num_labels)[target]  # One liner trick!


# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
train_X, test_X, train_Y, test_Y = train_test_split(all_X, all_Y, test_size=0.33, random_state=42)

# Step 1: Define the Model/Graph
X_size = train_X.shape[1]   # Number of input nodes: 4 features and 1 bias
Y_size = train_Y.shape[1]   # Number of outcomes (3 iris flowers)
X = tf.placeholder("float", shape=[1, X_size])
Y = tf.placeholder("float", shape=[1, Y_size])

W = tf.Variable(tf.random_normal([X_size,Y_size], stddev=0.1))

Yhat = tf.matmul(X,W)

# Step 2: Define Loss
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=Yhat))

# Step 3: Optimizer
optimizer = tf.train.GradientDescentOptimizer(0.001)
train = optimizer.minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# Step 4: Training Loop
for epoch in range(150):
        for i in range(len(train_X)):
            sess.run(train, feed_dict={X: train_X[i: i + 1], Y: train_Y[i: i + 1]})

# Step 5: Prediction
X = tf.constant([[1.,1.,1.,1.,1.]])
predict = tf.argmax(tf.matmul(X,W),axis=1)
print(sess.run(predict))
sess.close()




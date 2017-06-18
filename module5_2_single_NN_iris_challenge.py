# Module 5 Neural Network
# Challenge: One Layer NN with Iris Dataset

import tensorflow as tf
import numpy as np

n_nodes_hl1 = 100
n_classes = 3

# Step 1: Get Data
from sklearn import datasets
iris   = datasets.load_iris()
data   = iris["data"]
target = iris["target"]

# Step 2: Preprocess Data - One Hot Encoding
num_labels = len(np.unique(target))
all_Y = np.eye(num_labels)[target]  # One liner trick!

# Step 3: Shuffle/Spliut Data
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(data, all_Y, test_size=0.33, random_state=42)

# Step 4: Setup the Graph/Model
x_size = train_X.shape[1]   # Number of input nodes: 4 features and 1 bias
y_size = train_y.shape[1]

X = tf.placeholder("float", shape=[None, x_size])
y = tf.placeholder("float", shape=[None, y_size])


hidden_1_layer = {'weights': tf.Variable(tf.random_normal([4, n_nodes_hl1])),
                  'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_classes])),
                'biases': tf.Variable(tf.random_normal([n_classes]))}

l1 = tf.add(tf.matmul(X, hidden_1_layer['weights']), hidden_1_layer['biases'])
l1 = tf.nn.relu(l1)

output = tf.matmul(l1, output_layer['weights']) + output_layer['biases']
predict = tf.argmax(output, axis=1)

# Step 5 Loss Function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output))

# Step 6 Optimizer
optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(cost)

# Step 7 Training
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for epoch in range(70):
        # Train with each example
        for i in range(len(train_X)):
            sess.run(optimizer, feed_dict={X: train_X[i: i + 1], y: train_y[i: i + 1]})

        correct = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%"
              % (epoch + 1, 100. * accuracy.eval(feed_dict={X: train_X, y: train_y}), 100. * accuracy.eval(feed_dict={X: test_X, y: test_y})))

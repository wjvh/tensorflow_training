# Module 6 Convolutional Neural Network
# Challenge: Iris Dataset

import tensorflow as tf
import numpy as np

n_nodes_hl1 = 50
n_nodes_hl2 = 20
n_nodes_hl3 = 10

n_classes = 3

# Step 1: Get Data
from sklearn import datasets
iris = datasets.load_iris()
data = iris["data"]
target = iris["target"]

# Step 2: Preprocess Data - Convert to one-hot vectors
num_labels = len(np.unique(target))
all_Y = np.eye(num_labels)[target]  # One liner trick!

# Step 3: Shuffle/Split Data to Training/Test
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(data, all_Y, test_size=0.33, random_state=42)

x_size = train_X.shape[1]  # Number of input nodes: 4 features and 1 bias
y_size = train_y.shape[1]

# Step 4 Model/Graph
graph = tf.Graph()

with graph.as_default():
    X = tf.placeholder("float", shape=[None, x_size])
    y = tf.placeholder("float", shape=[None, y_size])

    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([x_size, n_nodes_hl1])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases': tf.Variable(tf.random_normal([n_classes])), }

    l1 = tf.add(tf.matmul(X, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

    predict = tf.argmax(output, axis=1)

    #Step 5: Loss Function
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output))
    
    #Step 6: Optimizer
    optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(cost)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    #Step 7 Training
    for epoch in range(15):
        # Train with each example
        for i in range(len(train_X)):
            sess.run(optimizer, feed_dict={X: train_X[i: i + 1], y: train_y[i: i + 1]})

        train_accuracy = np.mean(np.argmax(train_y, axis=1) ==
                                 sess.run(predict, feed_dict={X: train_X, y: train_y}))
        test_accuracy = np.mean(np.argmax(test_y, axis=1) ==
                                sess.run(predict, feed_dict={X: test_X, y: test_y}))

        print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%"
              % (epoch + 1, 100. * train_accuracy, 100. * test_accuracy))
    sess.close()

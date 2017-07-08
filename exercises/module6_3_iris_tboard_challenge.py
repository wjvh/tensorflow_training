# Module 6: Tensorboard
# Author: Dr. Alfred Ang
# Challange: Tensorbard for iris dataset

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Parameters
learning_rate = 0.005
training_epochs = 150
logdir = '/tmp/iris'

import tensorflow as tf
import numpy as np
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data
target = iris.target

# Convert the label into one-hot vector
num_labels = len(np.unique(target))
Y = np.eye(num_labels)[target]

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.33, random_state=42)

# Step 1: Initial Setup
with tf.name_scope('Inputs') as scope:
    X = tf.placeholder(tf.float32, [None, 4])
    y = tf.placeholder(tf.float32, [None, 3])

L1 = 200
L2 = 100
L3 = 60
L4 = 30

with tf.name_scope('Variables'):
    W1 = tf.Variable(tf.truncated_normal([4, L1], stddev=0.1))
    B1 = tf.Variable(tf.random_normal([L1]))
    W2 = tf.Variable(tf.truncated_normal([L1, L2], stddev=0.1))
    B2 = tf.Variable(tf.random_normal([L2]))
    W3 = tf.Variable(tf.truncated_normal([L2, L3], stddev=0.1))
    B3 = tf.Variable(tf.random_normal([L3]))
    W4 = tf.Variable(tf.truncated_normal([L3, L4], stddev=0.1))
    B4 = tf.Variable(tf.random_normal([L4]))
    W5 = tf.Variable(tf.truncated_normal([L4, 3], stddev=0.1))
    B5 = tf.Variable(tf.random_normal([3]))

# Step 2: Setup Model
with tf.name_scope('Model'):
    Y1 = tf.nn.relu(tf.matmul(X, W1) + B1)
    Y2 = tf.nn.relu(tf.matmul(Y1, W2) + B2)
    Y3 = tf.nn.relu(tf.matmul(Y2, W3) + B3)
    Y4 = tf.nn.relu(tf.matmul(Y3, W4) + B4)
    Ylogits = tf.matmul(Y4, W5) + B5
    yhat = tf.nn.softmax(Ylogits)

# Step 3: Loss Functions
with tf.name_scope('Loss'):
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=Ylogits))

# Step 4: Optimizer
with tf.name_scope('Train'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # optimizer = tf.train.AdamOptimizer(0.1)
    train = optimizer.minimize(loss)

# accuracy of the trained model, between 0 (worst) and 1 (best)
with tf.name_scope('Accuracy'):
    is_correct = tf.equal(tf.argmax(y,1),tf.argmax(yhat,1))
    accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Runnning the Graph on tensor board
writer = tf.summary.FileWriter(logdir, sess.graph)
tf.summary.scalar("Loss", loss)
tf.summary.scalar("Accuracy", accuracy)
summary_op = tf.summary.merge_all()

# Step 5: Training Loop
for epoch in range(training_epochs):
    for i in range(len(train_X)):
        train_data = {X: train_X[i: i + 1], y: train_Y[i: i + 1]}
        _, summary = sess.run([train, summary_op], feed_dict=train_data)
        writer.add_summary(summary, global_step=epoch*len(train_X)+i)
        print("Training Accuracy = ", sess.run(accuracy, feed_dict=train_data))

# Step 6: Evaluation
test_data = {X: test_X, y: test_Y}
print("Training Accuracy = ", sess.run(accuracy, feed_dict = test_data))

# Module 5: Neural Network and Deep Learning
# Tensorboard Challenge: NN model for iris dataset

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Parameters
learning_rate = 0.001
training_epochs = 150
logs_path = '/tmp/tensorflow/3'

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
    W = tf.Variable(tf.zeros([4, 3]))
    b = tf.Variable(tf.zeros([3]))

# Step 2: Define Model
with tf.name_scope('Model'):
    yhat = tf.matmul(X, W) + b
    y = tf.placeholder(tf.float32, [None, 3]) # Placeholder for correct answer

# Step 3: Loss Function
with tf.name_scope('Loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))

# Step 4: Optimizer
with tf.name_scope('Train'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train = optimizer.minimize(loss)

with tf.name_scope('Accuracy'):
    correct_prediction = tf.equal(tf.argmax(yhat, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.summary.scalar("Loss", loss)
tf.summary.scalar("Accuracy", accuracy)
summary_op = tf.summary.merge_all()

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

#Runnning the Graph on tensor board
file_writer = tf.summary.FileWriter(logs_path, sess.graph)


# Step 5: Training Loop
for epoch in range(training_epochs):
    for i in range(len(train_X)):
        training_data = {X: train_X[i: i + 1], y: train_Y[i: i + 1]}
        _, summary = sess.run([train, summary_op], feed_dict =training_data)
        file_writer.add_summary(summary, global_step=epoch*len(train_X) + i)
# Step 6: Evaluation
testing_data = {X: test_X, y: test_Y}
print("Training Accuracy = ", sess.run(accuracy, feed_dict = testing_data))

print("Run the command line")
print("tensorboard --logdir={}".format(logs_path))
print("Then open tensorboard on your web browser")
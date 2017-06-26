# Module 5: Neural Network and Deep Learning
# Tensorboard demo: NN model for MNIST dataset

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Parameters
learning_rate = 0.02
training_epochs = 2
batch_size = 100
logs_path = '/tmp/tensorflow/2'

import tensorflow as tf
tf.reset_default_graph()

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("mnist", one_hot=True)

# Step 1: Initial Setup
with tf.name_scope('Inputs') as scope:
    X = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])

L1 = 200
L2 = 100
L3 = 60
L4 = 30

with tf.name_scope('Variables'):
    W1 = tf.Variable(tf.truncated_normal([784, L1], stddev=0.1))
    B1 = tf.Variable(tf.zeros([L1]))
    W2 = tf.Variable(tf.truncated_normal([L1, L2], stddev=0.1))
    B2 = tf.Variable(tf.zeros([L2]))
    W3 = tf.Variable(tf.truncated_normal([L2, L3], stddev=0.1))
    B3 = tf.Variable(tf.zeros([L3]))
    W4 = tf.Variable(tf.truncated_normal([L3, L4], stddev=0.1))
    B4 = tf.Variable(tf.zeros([L4]))
    W5 = tf.Variable(tf.truncated_normal([L4, 10], stddev=0.1))
    B5 = tf.Variable(tf.zeros([10]))

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
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=y))

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

#Runnning the Graph on tensor board
file_writer = tf.summary.FileWriter(logs_path, sess.graph)
tf.summary.scalar("Loss", loss)
tf.summary.scalar("Accuracy", accuracy)
summary_op = tf.summary.merge_all()

# Step 5: Training Loop
for epoch in range(training_epochs):
    for i in range(int(55000/batch_size)):
        batch_X, batch_y = mnist.train.next_batch(batch_size)
        train_data = {X: batch_X, y: batch_y}
        _, summary = sess.run([train, summary_op], feed_dict =train_data)
        file_writer.add_summary(summary, global_step=epoch*int(55000/batch_size) + i)
        print("Training Accuracy = ", sess.run(accuracy, feed_dict=train_data))

# Step 6: Evaluation
test_data = {X:mnist.test.images,y:mnist.test.labels}
print("Testing Accuracy = ", sess.run(accuracy, feed_dict = test_data))

print("Run the command line")
print("tensorboard --logdir={}".format(logs_path))
print("Then open tensorboard on your web browser")
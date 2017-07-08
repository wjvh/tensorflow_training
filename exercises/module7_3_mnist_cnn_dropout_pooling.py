# Module 7: Convolutional Neural Network (CNN)
# CNN model with dropout for MNIST dataset

# CNN structure:
# · · · · · · · · · ·      input data                           X  [batch, 28, 28, 1]
# @ @ @ @ @ @ @ @ @ @   -- conv. layer 6x6x1x6 stride 1         W1 [6, 6, 1, 6]
# ∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶                                           Y1 [batch, 28, 28, 6]
#   @ @ @ @ @ @ @ @     -- conv. layer 5x5x6x12  stride 2       W2 [5, 5, 6, 12]
#   ∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶                                             Y2 [batch, 14, 14, 12]
#     @ @ @ @ @ @       -- conv. layer 4x4x12x24 stride 2       W3 [4, 4, 12, 24]
#     ∶∶∶∶∶∶∶∶∶∶∶                                               Y3 [batch, 7, 7, 24]
#      \x/x\x\x/        -- fully connected layer (relu)         W4 [7*7*24, 200]
#       · · · ·                                                 Y4 [batch, 200]
#       \x/x\x/         -- fully connected layer (softmax)      W5 [200, 10]
#        · · ·                                                  Y [batch, 10]

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Parameters
learning_rate = 0.01
training_epochs = 2
batch_size = 100

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("mnist", one_hot=True,reshape=False,validation_size=0)

# Step 1: Initial Setup
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
y = tf.placeholder(tf.float32, [None, 10])
pkeep = tf.placeholder(tf.float32)

# three convolutional layers with their channel counts, and a
# fully connected layer (tha last layer has 10 softmax neurons)
L1 = 6 # first convolutional layer output depth
L2 = 12 # second convolutional layer output depth
L3 = 24 # third convolutional layer
L4 = 200 # fully connected layer

W1 = tf.Variable(tf.truncated_normal([6,6,1,L1], stddev=0.1))
B1 = tf.Variable(tf.zeros([L1]))
W2 = tf.Variable(tf.truncated_normal([5,5,L1,L2], stddev=0.1))
B2 = tf.Variable(tf.zeros([L2]))
W3 = tf.Variable(tf.truncated_normal([4,4,L2,L3], stddev=0.1))
B3 = tf.Variable(tf.zeros([L3]))
W4 = tf.Variable(tf.truncated_normal([7*7*L3,L4], stddev=0.1))
B4 = tf.Variable(tf.zeros([L4]))
W5 = tf.Variable(tf.truncated_normal([L4, 10], stddev=0.1))
B5 = tf.Variable(tf.zeros([10]))

# Step 2: Setup Model
stride = 1  # output is 28x28
Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1,1,1,1], padding='SAME') + B1)
Y1 = tf.nn.max_pool(Y1, ksize=[1,2,2,1], strides=[1,stride,stride,1], padding='SAME')
stride = 2  # output is 14x14
Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1,1,1,1], padding='SAME') + B2)
Y2 = tf.nn.max_pool(Y2, ksize=[1,2,2,1], strides=[1,stride,stride,1], padding='SAME')
stride = 2  # output is 7x7
Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1,1,1,1], padding='SAME') + B3)
Y3 = tf.nn.max_pool(Y3, ksize=[1,2,2,1], strides=[1,stride,stride,1], padding='SAME')

# reshape the output from the third convolution for the fully connected layer
YY = tf.reshape(Y3, shape=[-1, 7 * 7 * L3])

Y4 = tf.nn.relu(tf.matmul(YY, W4) + B4)
YY4 = tf.nn.dropout(Y4, pkeep)
Ylogits = tf.matmul(Y4, W5) + B5
yhat = tf.nn.softmax(Ylogits)

# Step 3: Loss Functions
loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=y))

# Step 4: Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
# optimizer = tf.train.AdamOptimizer(0.1)
train = optimizer.minimize(loss)

# accuracy of the trained model, between 0 (worst) and 1 (best)
is_correct = tf.equal(tf.argmax(y,1),tf.argmax(yhat,1))
accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Step 5: Training Loop
for epoch in range(training_epochs):
    for i in range(int(mnist.train.num_examples/batch_size)):
        batch_X, batch_y = mnist.train.next_batch(batch_size)
        train_data = {X: batch_X, y: batch_y, pkeep: 1.0}
        sess.run(train, feed_dict=train_data)
        print("Training Accuracy = ", sess.run(accuracy, feed_dict=train_data))

# Step 6: Evaluation
test_data = {X:mnist.test.images,y:mnist.test.labels, pkeep: 1.0}
print("Testing Accuracy = ", sess.run(accuracy, feed_dict = test_data))
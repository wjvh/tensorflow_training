# Module 3: Datasets
# MNIST Handwriting Dataset

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("mnist", one_hot=True)

train_X = mnist.train.images
train_y = mnist.train.labels
test_X = mnist.test.images
test_y = mnist.test.labels

print(len(mnist.train.images))
print(len(mnist.test.images))
print(len(mnist.validation.images))

def show_digit(index):
    label = train_y[index].argmax(axis=0)
    image = train_X[index].reshape([28,28])
    plt.title('Digit : {}'.format(label))
    plt.imshow(image, cmap='gray_r')
    plt.show()

# show_digit(1)
# show_digit(2)
# show_digit(3)

batch_X, batch_Y = mnist.train.next_batch(100)
print(batch_X.shape)
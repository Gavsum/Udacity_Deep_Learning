### Model ###
# Image -> Conv -> MaxPool -> Conv ->
# MaxPool -> FullConnec -> FullConnec -> Classifier

# Load and one hot encode mnist data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(".", one_hot=True, reshape=False)

import tensorflow as tf

# Hyper Params
learning_rate = 0.00001
epochs = 10
batch_size = 128

# Number of samples to calculate validation / accuracy
test_valid_size = 256

# Network Parameters
n_classes = 10 # MNIST total classes 0 to 9
dropout = 0.75 # Probability to keep given units

# Store layer weights & Biases

weights = {
    'wc1': tf.Variable(tf.random_normal([5,5,1,32])),
    'wc2': tf.Variable(tf.random_normal([5,5,32,64])),
    'wd1': tf.Variable(tf.random_normal(7*7*64, 1024)),
    'out': tf.Variable(tf.random_normal(1024, n_classes))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Convolutional layer
def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)x

# Max Pooling Conv Layer
def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1,k,k,1], strides=[1,k,k,1], padding='SAME')
# Network
def conv_net(x, weights, biases, dropout):
    # Layer 1 (28*28*1) -> (14*14*32)
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    conv1 = maxpool2d(conv1, k=2)

    # Layer 2 (14*14*32) -> (7*7*64)
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv2 = maxpool2d(conv2, k=2)

    # Fully Connected Layer (7*7*64) -> (1024)
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output Layer/Class Prediction (1024) to (10)
    out = tf.add(tf.matmul(fc1, wights['out']), biases['out'])
    return out

# TF Graph inputs
x = tf.placeholder(tf.float32, [None, 28, 28, 1])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)

# Model
logits = conv_net(x, weights, biases, keep_prob)

# Define Loss and Optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Accuracy
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize Variables
init = tf.global_variables_initializer()

# Launch the Graph!

# Probably dont need this, is for getting cifar data
# from urllib.request import urlretrieve
from os.path import isfile, isdir
# Unittests python file in image_classification dir
import problem_unittests as tests
import numpy as np
import helper
import pdb as pdb
import pickle
import tensorflow as tf
from math import *

cifar_path = "cifar-10-batches-py"
tests.test_folder_path(cifar_path)


# Function to receive image data x, return as
# Normalized numpy array
# Values from 0 to 1 inclusive,
# Return shape same as x
def normalize(x):
    in_vals = np.array(x)
    divisor = np.full(in_vals.shape, 256)
    normed = np.divide(in_vals, divisor)

    return normed

# One Hot encoder
# input = list of labels,
# returns numpy array


def one_hot_encode(x):
    local = np.asarray(x)
    out_arr = np.empty([local.size, 10], dtype=object)

    for i, entry in enumerate(x):
        new_arr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        new_arr[entry] = 1
        out_arr[i] = new_arr

    return out_arr


# tests.test_normalize(normalize)
# tests.test_one_hot_encode(one_hot_encode)

# Pre-process data
#helper.preprocess_and_save_data(cifar_path, normalize, one_hot_encode)

# Checkpoint 1

# Loading Pre-processed data
valid_features, valid_labels = pickle.load(
    open('preprocess_validation.p', mode='rb'))


#### Input Functions ###

# Return a tensor for a batch of img input
def neural_net_image_input(image_shape):
    # Not sure if this is the best / most pythonic way to do this
    new_shape = [None]
    for p in image_shape:
        new_shape.append(p)

    return tf.placeholder(tf.float32, shape=new_shape, name='x')

# Return a Tensor for a batch of label inputs


def neural_net_label_input(n_classes):

    label_shape = [None]
    label_shape.append(n_classes)

    tens = tf.placeholder(tf.float32, shape=label_shape, name="y")

    return tens

# Return a Tensor for keep probability


def neural_net_keep_prob_input():
    keeper = tf.placeholder(tf.float32, name="keep_prob")
    return keeper


# Conv layer with max pooling
def conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides):
    # Create weight & bias using ksize, conv_outputs, & shape of x tensor

    # This may limit the shapes of input tensors that this function can accept (eg: pre defined this way)
    # Maybe recode to be more general
    new_shape = [
        conv_ksize[0],
        conv_ksize[1],
        x_tensor.shape[3].value,
        conv_num_outputs
    ]

    conv_format_stride = [
        1,
        conv_strides[0],
        conv_strides[1],
        1
    ]

    pool_format_stride = [
        1,
        pool_strides[0],
        pool_strides[1],
        1
    ]

    pool_format_ksize = [
        1,
        pool_ksize[0],
        pool_ksize[1],
        1
    ]

    weights = tf.Variable(tf.truncated_normal(new_shape))
    bias = tf.Variable(tf.zeros(conv_num_outputs))

    # Apply conv to x tensor using weights, stride & same padding
    conv_out = tf.nn.conv2d(x_tensor,
                            weights,
                            conv_format_stride,
                            padding="SAME")

    # Add bias
    conv_out = tf.nn.bias_add(conv_out, bias)

    # Add non linear activation
    conv_out = tf.nn.relu(conv_out)

    # Apply max pooling using pool size and strides + same padding
    conv_out = tf.nn.max_pool(conv_out,
                              pool_format_ksize,
                              pool_format_stride,
                              padding="SAME")

    return conv_out


def flatten(x_tensor):
    shape = x_tensor.get_shape().as_list()
    dimension = np.prod(shape[1:])
    flattened = tf.reshape(x_tensor, [-1, dimension])

    return flattened


def fully_conn(x_tensor, num_outputs):
    curr_shape = x_tensor.shape.as_list()
    weights = tf.Variable(tf.random_normal([curr_shape[1], num_outputs]))

    bias = tf.Variable(tf.zeros(num_outputs))

    fc = tf.add(tf.matmul(x_tensor, weights), bias)

    return tf.nn.relu(fc)


def output(x_tensor, num_outputs):
    curr_shape = x_tensor.shape.as_list()

    weights = tf.Variable(tf.random_normal([curr_shape[1], num_outputs]))

    bias = tf.Variable(tf.zeros(num_outputs))

    out = tf.add(tf.matmul(x_tensor, weights), bias)

    return out


def conv_net(x, keep_prob):
    """
    Create a convolutional neural network model
    : x: Placeholder tensor that holds image data.
    : keep_prob: Placeholder tensor that hold dropout keep probability.
    : return: Tensor that represents logits
    """
    # TODO: Apply 1, 2, or 3 Convolution and Max Pool layers
    #    Play around with different number of outputs, kernel size and stride
    # Function Definition from Above:
    #    conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides)
    conv1 = conv2d_maxpool(x_tensor = x,
                           conv_num_outputs = 50, 
                           conv_ksize = [2,2], 
                           conv_strides = [1,1], 
                           pool_ksize = [2,2], 
                           pool_strides = [1,1])
    

    # TODO: Apply a Flatten Layer
    # Function Definition from Above:
    #   flatten(x_tensor)
    flat = flatten(conv1)
    

    # TODO: Apply 1, 2, or 3 Fully Connected Layers
    #    Play around with different number of outputs
    # Function Definition from Above:
    #   fully_conn(x_tensor, num_outputs)
    fc = fully_conn(flat, 20)
    
    
    # TODO: Apply an Output Layer
    #    Set this to the number of classes
    # Function Definition from Above:
    #   output(x_tensor, num_outputs)
    out = output(fc, 10)
    
    
    # TODO: return output
    return out


def train_neural_network(session, optimizer, keep_probability, feature_batch, label_batch):
    """
    Optimize the session on a batch of images and labels
    : session: Current TensorFlow session
    : optimizer: TensorFlow optimizer function
    : keep_probability: keep probability
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    """
    # TODO: Implement Function
    session.run(optimizer, feed_dict={
        keep_prob: keep_probability,
        x: feature_batch,
        y: label_batch
        })

    pass

def print_stats(session, feature_batch, label_batch, cost, accuracy):
    """
    Print information about loss and validation accuracy
    : session: Current TensorFlow session
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    : cost: TensorFlow cost function
    : accuracy: TensorFlow accuracy function
    """
    



tf.reset_default_graph()
# tests.test_nn_image_inputs(neural_net_image_input)
# tests.test_nn_label_inputs(neural_net_label_input)
# tests.test_nn_keep_prob_inputs(neural_net_keep_prob_input)
# tests.test_con_pool(conv2d_maxpool)
# tests.test_flatten(flatten)
# tests.test_fully_conn(fully_conn)
# tests.test_output(output)
### ========== Additional model stuff ========== ###
# Inputs
x = neural_net_image_input((32, 32, 3))
y = neural_net_label_input(10)
keep_prob = neural_net_keep_prob_input()

# Model
logits = conv_net(x, keep_prob)

# Name logits Tensor, so that is can be loaded from disk after training
logits = tf.identity(logits, name='logits')

# Loss and Optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer().minimize(cost)

# Accuracy
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

### ========= End Additional model stuff ========= ###

#tests.test_conv_net(conv_net)
tests.test_train_nn(train_neural_network)
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
        new_arr = [0,0,0,0,0,0,0,0,0,0]
        new_arr[entry] = 1
        out_arr[i] = new_arr

    return out_arr


#tests.test_normalize(normalize)
#tests.test_one_hot_encode(one_hot_encode)

# Pre-process data 
#helper.preprocess_and_save_data(cifar_path, normalize, one_hot_encode)

# Checkpoint 1

# Loading Pre-processed data
valid_features, valid_labels = pickle.load(open('preprocess_validation.p', mode='rb'))


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
    new_shape =          [
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

    pool_format_ksize =  [
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
    """
    Apply a fully connected layer to x_tensor using weight and bias.
    x_tensor: A 2-D tensor where the first dimension is batch size.
    num_outputs: The number of output that the new tensor should be.
    return: A 2-D tensor where the second dimension is num_outputs
    """

tf.reset_default_graph()
# tests.test_nn_image_inputs(neural_net_image_input)
# tests.test_nn_label_inputs(neural_net_label_input)
# tests.test_nn_keep_prob_inputs(neural_net_keep_prob_input)
# tests.test_con_pool(conv2d_maxpool)
# tests.test_flatten(flatten)
test.test_fully_conn(fully_conn)


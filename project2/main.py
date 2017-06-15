# Probably dont need this, is for getting cifar data
# from urllib.request import urlretrieve
from os.path import isfile, isdir
# Unittests python file in image_classification dir
import problem_unittests as tests
import numpy as np
import helper
import pdb as pdb

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


tests.test_normalize(normalize)
tests.test_one_hot_encode(one_hot_encode)

# Pre-process data 
helper.preprocess_and_save_data(cifar_path, normalize, one_hot_encode)

# Checkpoint1



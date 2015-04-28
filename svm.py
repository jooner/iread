__author__ = 'trevorlutzow'

import os, struct
from array import array
import numpy as np
#from numba.decorators import jit, autojit

digit_images = os.path.join(".", 'digits/train-images')
digit_labels = os.path.join(".", 'digits/train-labels')

def load_mnist ():
  with open(digit_labels, 'rb') as f:
    labels = array("b", f.read())
  with open(digit_images, 'rb') as f:
    images = array("B", f.read())
  return images, labels

images, labels = load_mnist()

'''
 Each data point should be a column vector so that our data_array looks like:
    [ 1st     2nd     ...    last
      data    data    ...    data
      point   point   ...    point ]
'''
'''
 This will calculate pairwise distances by subtracting vectors so that the 
 output is:

    [ 1st-1st    1st-2nd    ...    1st-last
      2nd-1st    2nd-2nd    ...    2nd-last
      ...        ...        ...    ...-last
      last-1st   last-2nd   ...    last-last ]

 For example, get_euclidean_dist ([[1, 2],[3,4],[5,6]]) = 
        [[0, sqrt(3)],[sqrt(3), 0]] where our data points are:
    [1           [2
     3    and     4
     5]           6]
'''

def euclidean_dist (a, b):        
    return np.linalg.norm(a - b)

def dot_product (a, b):
    return np.dot(a, b)

def RBF (a, b, gamma):
    if gamma < 0:
        return np.exp(gamma * ((np.linalg.norm(a - b))**2))
    else:
        raise ValueError("gamma cannot be positive")    

def get_dist (data_array, kernel, gamma=None):
    columns = data_array.shape[1]
    output_matrix = np.zeros((columns, columns))
    for column1 in range(columns):
        for column2 in range(columns):
            if gamma:
                distance = kernel(data_array[:, column1], data_array[:, column2], gamma)
            else:
                distance = kernel(data_array[:, column1], data_array[:, column2])
            output_matrix[column1][column2] = distance
    return output_matrix

test_array = np.array([[1,2],[3,4],[5,6]])

print "\nEuclidean distance matrix of:\n", test_array, "\nis:\n"

print get_dist(test_array, euclidean_dist), "\n"

print "\nDot product distance matrix of:\n", test_array, "\nis:\n"

print get_dist(test_array, dot_product), "\n"

print "\nRBF distance matrix of:\n", test_array, "\nis:\n"

print get_dist(test_array, RBF, -10**(-2)), "\n"

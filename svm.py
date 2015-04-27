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

def subtraction (a, b):
    return a - b

def norm (a):
    total_distance = 0
    for element in a:
        total_distance = total_distance + element**2
    return (total_distance**0.5)

def get_euclidean_dist (data_array, single_distance_function, total_distance_function):
    
    def python_dist (data_array, single_distance_function, total_distance_function):
        rows = data_array.shape[0]        
        columns = data_array.shape[1]
        output_matrix = np.zeros((columns, columns))
        for column1 in range(columns):
            for column2 in range(columns):
                single_differences = []
                for row in range(rows):
                    single_differences.append(single_distance_function(
                            data_array[row][column1], data_array[row][column2]))
                total_distance = total_distance_function(single_differences)
                output_matrix[column1][column2] = total_distance
        return output_matrix

    distance_matrix = python_dist (data_array, single_distance_function, total_distance_function)
    
    return distance_matrix

test_array = np.array([[1,2],[3,4],[5,6]])

print "\nDistance matrix of:\n", test_array, "\nis:\n"

print get_euclidean_dist(test_array, subtraction, norm), "\n"











                    




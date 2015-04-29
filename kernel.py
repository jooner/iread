# -*- coding: utf-8 -*-

import numpy as np

class Kernel(object):
  """Handles different types of kernels"""

def euclidean_dist (a, b):        
  return np.linalg.norm(a - b)

def dot_product (a, b):
  return np.dot(a, b)

def RBF (a, b, gamma):
  if gamma < 0:
    return np.exp(gamma * ((np.linalg.norm(a - b))**2))
  else:
    raise ValueError("gamma cannot be positive")    

def get_dist (input_data_array, kernel, transpose=False, gamma=None):
  if transpose:
    data_array = np.transpose(input_data_array)
  else:
    data_array = input_data_array
  columns = data_array.shape[1]
  output_matrix = np.zeros((columns, columns))
  for column1 in range(columns):
    for column2 in range(columns):
      if gamma:
        # known problem --> use getattr() or global()/local()
        distance = euclidean_dist(data_array[:, column1], data_array[:, column2], gamma)
      else:
        distance = euclidean_dist(data_array[:, column1], data_array[:, column2])
      output_matrix[column1][column2] = distance
  if transpose:
    return np.transpose(output_matrix)
  else:
    return output_matrix

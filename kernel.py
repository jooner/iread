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

def get_dist (data_array, kernel, gamma=None):
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
  return output_matrix

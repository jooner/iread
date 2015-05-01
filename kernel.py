# -*- coding: utf-8 -*-

import numpy as np

def euclidean_dist (a, b):        
  return np.linalg.norm(a - b)

def dot_product (a, b):
  return np.dot(a, b)

def rbf (a, b, gamma):
  if gamma < 0:
    return np.exp(gamma * ((np.linalg.norm(a - b))**2))
  else:
    raise ValueError("Invalid Gamma")

def matrify(X, kernel, gamma=None):
  columns, _ = X.shape
  K = np.zeros((columns, columns))
  for i, x_i in enumerate(X):
    for j, x_j in enumerate(X):
      if kernel == "euclidean_dist":
        K[i, j] = euclidean_dist(x_i, x_j)
      elif kernel == "dot_product":
        K[i, j] = dot_product(x_i, x_j)
      else:
        K[i, j] = rbf(x_i, x_j, gamma)
  return K

def get_dist (input_data_array, kernel='rbf', transpose=False, gamma=0):

  '''if transpose:
    data_array = np.transpose(input_data_array)
  else:
    data_array = input_data_array
  columns = data_array.shape[1]
  output_matrix = np.zeros((columns, columns))'''
  for column1 in range(columns):
    for column2 in range(columns):
      if kernel == "euclidean_dist":
        distance = euclidean_dist(data_array[:, column1], data_array[:, column2])
      elif kernel == "dot_product":
        distance = dot_product(data_array[:, column1], data_array[:, column2])
      else:
        distance = np.exp(gamma * ((np.linalg.norm(a - b))**2))
        #distance = rbf(data_array[:, column1], data_array[:, column2], gamma)
      #output_matrix[column1][column2] = distance
  '''if transpose:
    return np.transpose(output_matrix)
  else:
    return output_matrix'''
  return distance

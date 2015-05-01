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

  for column1 in range(columns):
    for column2 in range(columns):
      if kernel == "euclidean_dist":
        distance = euclidean_dist(data_array[:, column1], data_array[:, column2])
      elif kernel == "dot_product":
        distance = dot_product(data_array[:, column1], data_array[:, column2])
      else:
        distance = np.exp(gamma * ((np.linalg.norm(a - b))**2))
  return distance


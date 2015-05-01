# -*- coding: utf-8 -*-

# libraries for svm module
from cvxopt import solvers, matrix

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from kernel import *
from time import time

# libraries for training and testing
import random
from sklearn.datasets import fetch_mldata

KERNEL = 'rbf'
GAMMA = -1e-5
RAND_SEED = 0
C_VAL = 1

class SVMTrain(object):
  """Implementation of SVM Classifier"""

  def __init__(self, kernel, gamma, transpose):
    self.kernel = kernel
    self.gamma = gamma
    self.transpose = transpose

  def lagrange_calc(self, X, y):
    num_samples, _ = X.shape
    K = matrify(X, self.kernel, self.gamma)
    P = matrix(np.outer(y, y) * K)
    q = matrix(-1 * np.ones(num_samples))
    G_pos = matrix(np.diag(np.ones(num_samples)))
    G_neg = matrix(np.diag(np.ones(num_samples)) * -1)
    G = matrix(np.vstack((G_neg, G_pos)))
    h_std = matrix(np.zeros(num_samples))
    h_slack = matrix(np.ones(num_samples))    
    h = matrix(np.vstack((h_std, h_slack)))
    A = matrix(list(y), (1, num_samples))
    b = matrix(0.0)
    # refer to quadratic programming in
    # http://cvxopt.org/userguide/coneprog.html#optional-solvers
    solution = solvers.qp(P, q, G, h, A, b)
    return np.ravel(solution['x'])

  def make_model(self, X, y, lagrange):
    
    supp_idx = []
    supp_mult = []
    supp_vectors = []
    supp_vector_labels = []
    for idx, val in enumerate(lagrange):
      if val > 1e-5:
        supp_idx.append(idx)
        supp_mult.append(val)
        supp_vectors.append(y[idx])
        supp_vector_labels.append(X[idx])
    supp_mult = np.array(supp_mult)
    supp_vectors = np.array(supp_vectors)
    supp_vector_labels = np.array(supp_vector_labels)

    """
    supp_idx = lagrange > 1e-5
    supp_mult = lagrange[supp_idx]

    supp_vectors, supp_vector_labels = X[supp_idx], y[supp_idx]
    """

    bias = np.mean([y_k - SVMTest(kernel=self.kernel,
                                  bias=0.0,
                                  weights=supp_mult,
                                  supp_vectors=supp_vectors,
                                  transpose=self.transpose,
                                  supp_vector_labels=supp_vector_labels
                                  ).predict(x_k)
    for (y_k, x_k) in zip(supp_vector_labels, supp_vectors)])
    return SVMTest(kernel=self.kernel, bias=bias, weights=supp_mult,
                   supp_vectors=supp_vectors, transpose=self.transpose,
                   supp_vector_labels=supp_vector_labels)

  def train(self, X, y):
    return self.make_model(X, y, self.lagrange_calc(X, y))

class SVMTest(object):
  def __init__(self, kernel, bias, weights, supp_vectors,
               transpose, supp_vector_labels):
    self.kernel = kernel
    self.bias = bias
    self.weights = weights
    self.supp_vectors = supp_vectors
    self.transpose = transpose
    self.supp_vector_labels = supp_vector_labels

  def predict(self, x):
    """Computes the SVM prediction on the given features x"""
    result = self.bias
    if self.kernel == "euclidean_dist":
      kernel = lambda x, y: euclidean_dist(x, y)
    elif self.kernel == "dot_product":
      kernel = lambda x, y: dot_product(x, y)
    else:
      kernel = lambda x, y, z: rbf(x, y, z)
    for z_i, x_i, y_i in zip(self.weights,
                             self.supp_vectors,
                             self.supp_vector_labels):
        result += z_i * y_i * kernel(x_i, x, GAMMA)
    return np.sign(result)

def main():
  # Default is True
  transpose = True
  mnist = fetch_mldata('MNIST original', transpose_data = transpose)
  # Truncate the data
  n_train = 40000
  n_test = 1000
  # Split training and testing sets
  indices = np.arange(len(mnist.data))
  random.seed(RAND_SEED)
  train_idx = random.sample(indices, n_train)
  test_idx = random.sample(indices, n_test)
  X_train, y_train = mnist.data[train_idx], mnist.target[train_idx]
  X_test, y_test = mnist.data[test_idx], mnist.target[test_idx]
  clf = SVMTrain(KERNEL, GAMMA, transpose).train(X_train, y_train)
  y_pred = clf.predict(X_test)
  print y_pred, y_test



if __name__ == "__main__":
  start_time = time()
  main()
  print "Runtime: ", time() - start_time

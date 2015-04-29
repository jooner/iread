# -*- coding: utf-8 -*-

from __future__ import print_function

# libraries for svm module
from cvxopt import solvers, matrix

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import kernel
from time import time

# libraries for training and testing
import random
from sklearn import cross_validation, metrics
from sklearn.datasets import fetch_mldata

KERNEL = 'RBF'
GAMMA = 1e-5
RAND_SEED = 0
C_VAL = 1

class SVMTrain(object):
  """Implementation of SVM Classifier"""

  def __init__(self, kernel, gamma, C):
    self.kernel = kernel
    self.gamma = gamma
    self.C = C

  def lagrange_calc(self, X, y):
    num_samples, _ = X.shape
    K = kernel.get_dist(X, KERNEL, gamma=None)
    P = matrix(np.outer(y, y) * K)
    q = matrix(-1 * np.ones(num_samples))
    G_std = matrix(np.diag(np.ones(num_samples) * -1))
    h_std = matrix(np.zeros(num_samples))
    G_slack = matrix(np.diag(np.ones(num_samples)))
    h_slack = matrix(np.ones(num_samples) * self.C)
    # vertically stack
    G = matrix(np.vstack((G_std, G_slack)))
    h = matrix(np.vstack((h_std, h_slack)))
    A = matrix(y, (1, num_samples))
    b = matrix(0.0)
    # refer to quadratic programming in
    # http://cvxopt.org/userguide/coneprog.html#optional-solvers
    solution = solvers.qp(P, q, G, h, A, b)
    return np.ravel(solution['x'])

  def make_model(X, y, lagrange):
    supp_mult = lagrange[supp_idx]
    supp_vectors, supp_vector_labels = X[supp_idx], y[supp_idx]
    bias = np.mean([y_k - SVMTest(kernel=self._kernel,
                                       bias=0.0,
                                       weights=supp_mult,
                                       supp_vectors=supp_vectors,
                                       supp_vector_labels=supp_vector_labels
                                  ).predict(x_k)
    for (y_k, x_k) in zip(supp_vector_labels, supp_vectors)])
    return SVMTest(kernel=self.kernel, bias=bias, weights=supp_mult,
                   supp_vectors=support_vectors,
                   supp_vector_labels=supp_vector_labels)

  def train(self, X, y):
    return self.make_model(X, y, self.lagrange_calc(X, y))

class SVMTest(object):
  def __init__(self, kernel, bias, weights, supp_vectors, supp_vector_labels):
    self.kernel = kernel
    self.bias = bias
    self.weights = weights
    self.supp_vectors = supp_vectors
    self.supp_vector_labels = supp_vector_labels

  def predict(self, x):
    """Computes the SVM prediction on the given features x"""
    result = self.bias
    for z_i, x_i, y_i in zip(self.weights,
                             self.supp_vectors,
                             self.supp_vector_labels):
        result += z_i * y_i * self.kernel(x_i, x)
    return np.sign(result).item()

def main():
  mnist = fetch_mldata('MNIST original')
  # Trunk the data
  n_train = 500
  n_test = 20
  # Split training and testing sets
  indices = np.arange(len(mnist.data))
  random.seed(RAND_SEED)
  train_idx = random.sample(indices, n_train)
  test_idx = random.sample(indices, n_test)
  X_train, y_train = mnist.data[train_idx], mnist.target[train_idx]
  X_test, y_test = mnist.data[test_idx], mnist.target[test_idx]
  print (X_train, y_train, X_test, y_test)
  clf = SVMTrain(KERNEL, GAMMA, C_VAL).train(X_train, y_train)
  y_pred = clf.predict(X_test)
  print (y_pred, y_test)

if __name__ == "__main__":
  start_time = time()
  main()
  print ("Runtime: ", time() - start_time)

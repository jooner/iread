# -*- coding: utf-8 -*-

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

KERNEL = 'euclidian_dist'
GAMMA = -1e-5
RAND_SEED = 0
C_VAL = 1

class SVMTrain(object):
  """Implementation of SVM Classifier"""

  def __init__(self, kernel, gamma, C, transpose):
    self.kernel = kernel
    self.gamma = gamma
    self.C = C
    self.transpose = transpose

  def lagrange_calc(self, X, y):
    num_samples, _ = X.shape
    print "num_samples : ", num_samples
    K = kernel.get_dist(X, kernel=KERNEL, transpose=self.transpose, gamma=GAMMA)
    P = matrix(np.outer(y, y) * K)
    q = matrix(-1 * np.ones(num_samples))
    G_pos = matrix(np.diag(np.ones(num_samples)))
    G_neg = matrix(np.diag(np.ones(num_samples)) * -1)
    G = matrix(np.vstack((G_neg, G_pos)))
    h_std = matrix(np.zeros(num_samples))
    h_slack = matrix(np.ones(num_samples) * self.C)    
    h = matrix(np.vstack((h_std, h_slack)))
    A = matrix(list(y), (1, num_samples))
    b = matrix(0.0)
    # refer to quadratic programming in
    # http://cvxopt.org/userguide/coneprog.html#optional-solvers
    solution = solvers.qp(P, q, G, h, A, b)
    return np.ravel(solution['x'])

  def make_model(self, X, y, lagrange):
    supp_idx = lagrange > 10e-5
    supp_mult = lagrange[supp_idx]
    supp_vectors, supp_vector_labels = X[supp_idx], y[supp_idx]
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
    for z_i, x_i, y_i in zip(self.weights,
                             self.supp_vectors,
                             self.supp_vector_labels):
        result += z_i * y_i * kernel.get_dist(x, kernel=KERNEL,
                              transpose=self.transpose, gamma=GAMMA)
    return np.sign(result).item()

def main():
  ## Default is True
  transpose = True
  mnist = fetch_mldata('MNIST original', transpose_data = transpose)
  # Trunk the data
  n_train = 500
  n_test = 10
  # Split training and testing sets
  indices = np.arange(len(mnist.data))
  random.seed(RAND_SEED)
  train_idx = random.sample(indices, n_train)
  test_idx = random.sample(indices, n_test)
  X_train, y_train = mnist.data[train_idx], mnist.target[train_idx]
  X_test, y_test = mnist.data[test_idx], mnist.target[test_idx]
  clf = SVMTrain(KERNEL, GAMMA, C_VAL, transpose).train(X_train, y_train)
  y_pred = clf.predict(X_test)
  print y_pred, y_test


if __name__ == "__main__":
  start_time = time()
  main()
  print "Runtime: ", time() - start_time

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

"""KERNEL = 'rbf'"""
GAMMA = -1e-5
RAND_SEED = 0
C_VAL = 1
KERNEL = lambda x, y, z: rbf(x, y, z)

class SVMTrain(object):
  """Implementation of SVM Classifier"""

  def __init__(self, kernel, gamma):
    self.kernel = kernel
    self.gamma = gamma

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
    #print P.size
    #print q.size
    #print G.size
    #print h.size
    #print A.size
    #print b.size
    # refer to quadratic programming in
    # http://cvxopt.org/userguide/coneprog.html#optional-solvers
    solution = solvers.qp(P, q, G, h, A, b)
    #print solution
    #print np.ravel(solution['x'])
    return np.ravel(solution['x'])

  def make_model(self, X, y, lagrange):
    print "y labels: {}".format(y)
    #supp_idx = []
    supp_mult = []
    supp_vectors = []
    supp_vector_labels = []
    for idx, val in enumerate(lagrange):
      if val:
        supp_mult.append(val)
        supp_vectors.append(X[idx])
        supp_vector_labels.append(y[idx])
    supp_mult = np.array(supp_mult)
    supp_vectors = np.array(supp_vectors)
    supp_vector_labels = np.array(supp_vector_labels)
    '''
    support_idx = \
            lagrange > 1e-5

    
    supp_idx = lagrange > 1e-5
    supp_mult = lagrange[supp_idx]

    supp_vectors, supp_vector_labels = X[supp_idx], y[supp_idx]
    '''

    bias = np.mean([y_k - SVMTest(kernel=self.kernel,
                                  bias=0.0,
                                  weights=supp_mult,
                                  supp_vectors=supp_vectors,
                                  supp_vector_labels=supp_vector_labels
                                  ).predict(x_k)
    for (y_k, x_k) in zip(supp_vector_labels, supp_vectors)])
    print "bias: {}".format(bias)
    return SVMTest(kernel=self.kernel, bias=bias, weights=supp_mult,
                   supp_vectors=supp_vectors, supp_vector_labels=supp_vector_labels)

  def train(self, X, y):
    return self.make_model(X, y, self.lagrange_calc(X, y))

class SVMTest(object):
  def __init__(self, kernel, bias, weights, supp_vectors,
               supp_vector_labels):
    self.kernel = kernel
    self.bias = bias
    self.weights = weights
    self.supp_vectors = supp_vectors
    self.supp_vector_labels = supp_vector_labels
    assert len(supp_vectors) == len(supp_vector_labels)
    assert len(weights) == len(supp_vector_labels)

  def predict(self, x):
    """Computes the SVM prediction on the given features x"""
    result = self.bias
    #print "x.shape before : ", x.shape
    #print self.supp_vectors.shape
    for z_i, x_i, y_i in zip(self.weights,
                             self.supp_vectors,
                             self.supp_vector_labels):
        result += z_i * y_i * self.kernel(x, x_i, GAMMA)
    #print "z_i: {} x_i: {} y_i {}".format(z_i, x_i, y_i)
    #print "x.shape :", x.shape
    #print "result here :", result

    return np.sign(result)

def main():
  mnist = fetch_mldata('MNIST original')
  # Truncate the data
  n_train = 400
  n_test = 10
  # Split training and testing sets
  indices = np.arange(len(mnist.data))
  random.seed(RAND_SEED)
  train_idx = random.sample(indices, n_train)
  test_idx = random.sample(indices, n_test)
  X_train, y_train = mnist.data[train_idx], mnist.target[train_idx]
  print "y_train :",y_train
  X_test, y_test = mnist.data[test_idx], mnist.target[test_idx]
  clf = SVMTrain(KERNEL, GAMMA).train(X_train, y_train)
  y_pred = []
  print "length x_test :", len(X_test)
  ##Now it spits out the result of our function--greater than or less than one
  for test_data in X_test:
    y_pred.append((clf.predict(test_data.reshape(1,784))).tolist())
  print X_test.shape
  print y_test.shape
  print X_train.shape
  print y_train.shape
  #print y_pred.shape
  print y_pred, y_test



if __name__ == "__main__":
  start_time = time()
  main()
  print "Runtime: ", time() - start_time

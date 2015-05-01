# -*- coding: utf-8 -*-

# libraries for svm module
from cvxopt import solvers, matrix

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from kernel import *
from time import time
from math import sin, cos

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import itertools

# libraries for training and testing
import random
from sklearn import cross_validation, metrics
from sklearn.datasets import fetch_mldata

KERNEL = 'rbf'
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
    A = matrix(y, (1, num_samples))
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
      ## since support vectors will have lagrangians > 0
      if val > 1e-5:
        supp_idx.append(idx)
        supp_mult.append(val)
        supp_vectors.append(X[idx])
        supp_vector_labels.append(y[idx])
    supp_mult = np.array(supp_mult)
    supp_vectors = np.array(supp_vectors)
    supp_vector_labels = np.array(supp_vector_labels)
    bias = np.mean([y_k - SVMTest(kernel=self.kernel,
                                  bias=0.0,
                                  weights=supp_mult,
                                  supp_vectors=supp_vectors,
                                  supp_vector_labels=supp_vector_labels
                                  ).predict(x_k)
    for (y_k, x_k) in zip(supp_vector_labels, supp_vectors)])
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

  def predict(self, x):
    """Computes the SVM prediction on the given features x"""
    result = self.bias
    for z_i, x_i, y_i in zip(self.weights,
                             self.supp_vectors,
                             self.supp_vector_labels):
        result += z_i * y_i * self.kernel(x, x_i, GAMMA)
    #print "result :", result

    return np.sign(result).item()

def main():
  mnist = fetch_mldata('MNIST original')
  # Truncate the data
  n_train = 1000
  ## Some n_train random numbers that we will test on in 2-D.
  X_train = np.empty([n_train, 2])
  for i in xrange(n_train):
    X_train[i][0], X_train[i][1] = random.uniform(-5, 5), random.uniform(-5, 5)
  ## classify points as either above or below the line
  y_train = []
  for el in X_train:
    if el[0] - el[1] < 0:
      y_train.append(1.0)
    else:
      y_train.append(-1.0)
  y_train = np.matrix(y_train).reshape(n_train, 1)
  clf = SVMTrain(KERNEL, GAMMA).train(X_train, y_train)
  plot(clf, X_train, y_train, 20, 'test.pdf')


def plot(predictor, X, y, grid_size, filename):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_size),
                         np.linspace(y_min, y_max, grid_size),
                         indexing='ij')
    results = []
    points = []
    for (i, j) in itertools.product(range(grid_size), range(grid_size)):
        point = np.array([xx[i, j], yy[i, j]]).reshape(1, 2)
        results.append(predictor.predict(point))
        points.append((xx[i,j], yy[i,j]))
    Z = np.array(results).reshape(xx.shape)
    Z_summed = np.sum(Z, axis = 1)
    with open ("results.txt", "w") as f:
        for i, result in enumerate(results):
            f.write("{} is classified as {}\n".format(points[i], result))
    plt.contourf(xx, yy, Z,
                 cmap=cm.Paired,
                 levels=[-0.001, 0.001],
                 extend='both',
                 alpha=0.8)
    plt.scatter(np.array(X[:, 0]).reshape(-1,), np.array(X[:, 1]).reshape(-1,),
                c=np.array(y).reshape(-1,), cmap=cm.Paired)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.savefig(filename)

if __name__ == "__main__":
  start_time = time()
  main()
  print "Runtime: ", time() - start_time


#from numpy import *
#from sklearn import cross_validation
#from sklearn.svm import SVC
#from sklearn.datasets import fetch_mldata
#import os
#
#""" Loading data """
#mnist = fetch_mldata("MNIST original", data_home=os.path.join(".", 'digits'))
#X, Y = mnist.data, mnist.target
#
#""" Rescale grayscale from -1 to 1 """
#X = X/255.0*2 - 1
#
#""" Shuffle the input """
#shuffle = random.permutation(arange(X.shape[0]))
#X, Y = X[shuffle], Y[shuffle]
#
#""" Initialise the model"""
#clf = SVC(kernel="rbf", C=2.8, gamma=.0073)
#
#""" Train and validate the model with n-fold cross validation """
#scores = cross_validation.cross_val_score(clf, X, Y, cv=2)
#
#print scores

import numpy
import random
from numpy import arange
#from classification import *
from sklearn import cross_validation
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.datasets import fetch_mldata
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
import time
"""
def run():
    mnist = fetch_mldata('MNIST original')
    #mnist.data, mnist.target = shuffle(mnist.data, mnist.target)
    #print mnist.data.shape
    # Trunk the data
    print dir(mnist.data.dot)
    print dir(mnist.data)
    n_train = 5000
    n_test = 200

    # Define training and testing sets
    indices = arange(len(mnist.data))
    random.seed(0)
    train_idx = random.sample(indices, n_train)
    test_idx = random.sample(indices, n_test)
    #train_idx = arange(0,n_train)
    #test_idx = arange(n_train+1,n_train+n_test)
 
    X_train, y_train = mnist.data[train_idx], mnist.target[train_idx]
    X_test, y_test = mnist.data[test_idx], mnist.target[test_idx]
    print X_train, y_train, X_test, y_test

    # Apply a learning algorithm
    print "Applying a learning algorithm..."
    #clf = RandomForestClassifier(n_estimators=10,n_jobs=2)
    clf = SVC(kernel="rbf", C=1, gamma=.0001)
    scores = cross_validation.cross_val_score(clf, X_train, y_train, cv=3)
    print scores
    clf.fit(X_train, y_train)
	 
    # Make a prediction
    print "Making predictions..."
    y_pred = clf.predict(X_test)
 
    print y_pred
 
    #Evaluate the prediction
    print "Evaluating results..."
    print "Precision:", metrics.precision_score(y_test, y_pred)
    print "Recall:", metrics.recall_score(y_test, y_pred)
    print "F1 score:", metrics.f1_score(y_test, y_pred)
    print "Mean accuracy:", clf.score(X_test, y_test)
"""
 
if __name__ == "__main__":
    start_time = time.time()
    results = run()
    end_time = time.time()
    print "Overall running time:", end_time - start_time



#######################

class SVMTrainer(object):
    def __init__(self, kernel, c):
        self._kernel = kernel
        self._c = c

    def train(self, X, y):
        """Given the training features X with labels y, returns a SVM
        predictor representing the trained SVM.
        """
        lagrange_multipliers = self._compute_multipliers(X, y)
        return self._construct_predictor(X, y, lagrange_multipliers)

    def _gram_matrix(self, X):
        n_samples, n_features = X.shape
        K = np.zeros((n_samples, n_samples))
        # TODO(tulloch) - vectorize
        for i, x_i in enumerate(X):
            for j, x_j in enumerate(X):
                K[i, j] = self._kernel(x_i, x_j)
        return K

    def _construct_predictor(self, X, y, lagrange_multipliers):
        support_vector_indices = \
            lagrange_multipliers > MIN_SUPPORT_VECTOR_MULTIPLIER

        support_multipliers = lagrange_multipliers[support_vector_indices]
        support_vectors = X[support_vector_indices]
        support_vector_labels = y[support_vector_indices]

        # http://www.cs.cmu.edu/~guestrin/Class/10701-S07/Slides/kernels.pdf
        # bias = y_k - \sum z_i y_i  K(x_k, x_i)
        # Thus we can just predict an example with bias of zero, and
        # compute error.
        bias = np.mean(
            [y_k - SVMPredictor(
                kernel=self._kernel,
                bias=0.0,
                weights=support_multipliers,
                support_vectors=support_vectors,
                support_vector_labels=support_vector_labels).predict(x_k)
             for (y_k, x_k) in zip(support_vector_labels, support_vectors)])

        return SVMPredictor(
            kernel=self._kernel,
            bias=bias,
            weights=support_multipliers,
            support_vectors=support_vectors,
            support_vector_labels=support_vector_labels)

    def _compute_multipliers(self, X, y):
        n_samples, n_features = X.shape

        K = self._gram_matrix(X)
        # Solves
        # min 1/2 x^T P x + q^T x
        # s.t.
        #  Gx \coneleq h
        #  Ax = b

        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(-1 * np.ones(n_samples))

        # -a_i \leq 0
        # TODO(tulloch) - modify G, h so that we have a soft-margin classifier
        G_std = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
        h_std = cvxopt.matrix(np.zeros(n_samples))

        # a_i \leq c
        G_slack = cvxopt.matrix(np.diag(np.ones(n_samples)))
        h_slack = cvxopt.matrix(np.ones(n_samples) * self._c)

        G = cvxopt.matrix(np.vstack((G_std, G_slack)))
        h = cvxopt.matrix(np.vstack((h_std, h_slack)))

        A = cvxopt.matrix(y, (1, n_samples))
        b = cvxopt.matrix(0.0)

        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        return np.ravel(solution['x'])


class SVMPredictor(object):
    def __init__(self,
                 kernel,
                 bias,
                 weights,
                 support_vectors,
                 support_vector_labels):
        self._kernel = kernel
        self._bias = bias
        self._weights = weights
        self._support_vectors = support_vectors
        self._support_vector_labels = support_vector_labels
        assert len(support_vectors) == len(support_vector_labels)
        assert len(weights) == len(support_vector_labels)
        logging.info("Bias: %s", self._bias)
        logging.info("Weights: %s", self._weights)
        logging.info("Support vectors: %s", self._support_vectors)
        logging.info("Support vector labels: %s", self._support_vector_labels)

    def predict(self, x):
        """
        Computes the SVM prediction on the given features x.
        """
        result = self._bias
        for z_i, x_i, y_i in zip(self._weights,
                                 self._support_vectors,
                                 self._support_vector_labels):
            result += z_i * y_i * self._kernel(x_i, x)
        return np.sign(result).item()

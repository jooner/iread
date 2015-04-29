class BaseLibSVM(six.with_metaclass(ABCMeta, BaseEstimator)):

  """Base class for estimators that use libsvm as backing library
  This implements support vector machine classification and regression.
  Parameter documentation is in the derived `SVC` class.
  """

  # The order of these must match the integer values in LibSVM.
  # XXX These are actually the same in the dense case. Need to factor
  # this out.
  _sparse_kernels = ["linear", "poly", "rbf", "sigmoid", "precomputed"]

  @abstractmethod
  def __init__(self, impl, kernel, degree, gamma, coef0,
               tol, C, nu, epsilon, shrinking, probability, cache_size,
               class_weight, verbose, max_iter, random_state):

      if impl not in LIBSVM_IMPL:  # pragma: no cover
          raise ValueError("impl should be one of %s, %s was given" % (
              LIBSVM_IMPL, impl))

      self._impl = impl
      self.kernel = kernel
      self.degree = degree
      self.gamma = gamma
      self.coef0 = coef0
      self.tol = tol
      self.C = C
      self.nu = nu
      self.epsilon = epsilon
      self.shrinking = shrinking
      self.probability = probability
      self.cache_size = cache_size
      self.class_weight = class_weight
      self.verbose = verbose
      self.max_iter = max_iter
      self.random_state = random_state

  @property
  def _pairwise(self):
      # Used by cross_val_score.
      kernel = self.kernel
      return kernel == "precomputed" or callable(kernel)

  def fit(self, X, y, sample_weight=None):
      """Fit the SVM model according to the given training data.
      Parameters
      ----------
      X : {array-like, sparse matrix}, shape (n_samples, n_features)
          Training vectors, where n_samples is the number of samples
          and n_features is the number of features.
          For kernel="precomputed", the expected shape of X is
          (n_samples, n_samples).
      y : array-like, shape (n_samples,)
          Target values (class labels in classification, real numbers in
          regression)
      sample_weight : array-like, shape (n_samples,)
          Per-sample weights. Rescale C per sample. Higher weights
          force the classifier to put more emphasis on these points.
      Returns
      -------
      self : object
          Returns self.
      Notes
      ------
      If X and y are not C-ordered and contiguous arrays of np.float64 and
      X is not a scipy.sparse.csr_matrix, X and/or y may be copied.
      If X is a dense array, then the other methods will not support sparse
      matrices as input.
      """

      rnd = check_random_state(self.random_state)

      sparse = sp.isspmatrix(X)
      if sparse and self.kernel == "precomputed":
          raise TypeError("Sparse precomputed kernels are not supported.")
      self._sparse = sparse and not callable(self.kernel)

      X = check_array(X, accept_sparse='csr', dtype=np.float64, order='C')
      y = self._validate_targets(y)

      sample_weight = np.asarray([]
                                 if sample_weight is None
                                 else sample_weight, dtype=np.float64)
      solver_type = LIBSVM_IMPL.index(self._impl)

      # input validation
      if solver_type != 2 and X.shape[0] != y.shape[0]:
          raise ValueError("X and y have incompatible shapes.\n" +
                           "X has %s samples, but y has %s." %
                           (X.shape[0], y.shape[0]))

      if self.kernel == "precomputed" and X.shape[0] != X.shape[1]:
          raise ValueError("X.shape[0] should be equal to X.shape[1]")

      if sample_weight.shape[0] > 0 and sample_weight.shape[0] != X.shape[0]:
          raise ValueError("sample_weight and X have incompatible shapes: "
                           "%r vs %r\n"
                           "Note: Sparse matrices cannot be indexed w/"
                           "boolean masks (use `indices=True` in CV)."
                           % (sample_weight.shape, X.shape))

      if (self.kernel in ['poly', 'rbf']) and (self.gamma == 0):
          # if custom gamma is not provided ...
          self._gamma = 1.0 / X.shape[1]
      else:
          self._gamma = self.gamma

      kernel = self.kernel
      if callable(kernel):
          kernel = 'precomputed'

      fit = self._sparse_fit if self._sparse else self._dense_fit
      if self.verbose:  # pragma: no cover
          print('[LibSVM]', end='')

      seed = rnd.randint(np.iinfo('i').max)
      fit(X, y, sample_weight, solver_type, kernel, random_seed=seed)
      # see comment on the other call to np.iinfo in this file

      self.shape_fit_ = X.shape

      # In binary case, we need to flip the sign of coef, intercept and
      # decision function. Use self._intercept_ and self._dual_coef_ internally.
      self._intercept_ = self.intercept_.copy()
      self._dual_coef_ = self.dual_coef_
      if self._impl in ['c_svc', 'nu_svc'] and len(self.classes_) == 2:
          self.intercept_ *= -1
          self.dual_coef_ = -self.dual_coef_

      return self

  def _validate_targets(self, y):
      """Validation of y and class_weight.
      Default implementation for SVR and one-class; overridden in BaseSVC.
      """
      # XXX this is ugly.
      # Regression models should not have a class_weight_ attribute.
      self.class_weight_ = np.empty(0)
      return np.asarray(y, dtype=np.float64, order='C')

  def _warn_from_fit_status(self):
      assert self.fit_status_ in (0, 1)
      if self.fit_status_ == 1:
          warnings.warn('Solver terminated early (max_iter=%i).'
                        '  Consider pre-processing your data with'
                        ' StandardScaler or MinMaxScaler.'
                        % self.max_iter, ConvergenceWarning)

  def _dense_fit(self, X, y, sample_weight, solver_type, kernel,
                 random_seed):
      if callable(self.kernel):
          # you must store a reference to X to compute the kernel in predict
          # TODO: add keyword copy to copy on demand
          self.__Xfit = X
          X = self._compute_kernel(X)

          if X.shape[0] != X.shape[1]:
              raise ValueError("X.shape[0] should be equal to X.shape[1]")

      libsvm.set_verbosity_wrap(self.verbose)

      # we don't pass **self.get_params() to allow subclasses to
      # add other parameters to __init__
      self.support_, self.support_vectors_, self.n_support_, \
          self.dual_coef_, self.intercept_, self.probA_, \
          self.probB_, self.fit_status_ = libsvm.fit(
              X, y,
              svm_type=solver_type, sample_weight=sample_weight,
              class_weight=self.class_weight_, kernel=kernel, C=self.C,
              nu=self.nu, probability=self.probability, degree=self.degree,
              shrinking=self.shrinking, tol=self.tol,
              cache_size=self.cache_size, coef0=self.coef0,
              gamma=self._gamma, epsilon=self.epsilon,
              max_iter=self.max_iter, random_seed=random_seed)

      self._warn_from_fit_status()

  def _sparse_fit(self, X, y, sample_weight, solver_type, kernel,
                  random_seed):
      X.data = np.asarray(X.data, dtype=np.float64, order='C')
      X.sort_indices()

      kernel_type = self._sparse_kernels.index(kernel)

      libsvm_sparse.set_verbosity_wrap(self.verbose)

      self.support_, self.support_vectors_, dual_coef_data, \
          self.intercept_, self.n_support_, \
          self.probA_, self.probB_, self.fit_status_ = \
          libsvm_sparse.libsvm_sparse_train(
              X.shape[1], X.data, X.indices, X.indptr, y, solver_type,
              kernel_type, self.degree, self._gamma, self.coef0, self.tol,
              self.C, self.class_weight_,
              sample_weight, self.nu, self.cache_size, self.epsilon,
              int(self.shrinking), int(self.probability), self.max_iter,
              random_seed)

      self._warn_from_fit_status()

      if hasattr(self, "classes_"):
          n_class = len(self.classes_) - 1
      else:   # regression
          n_class = 1
      n_SV = self.support_vectors_.shape[0]

      dual_coef_indices = np.tile(np.arange(n_SV), n_class)
      dual_coef_indptr = np.arange(0, dual_coef_indices.size + 1,
                                   dual_coef_indices.size / n_class)
      self.dual_coef_ = sp.csr_matrix(
          (dual_coef_data, dual_coef_indices, dual_coef_indptr),
          (n_class, n_SV))

  def predict(self, X):
      """Perform regression on samples in X.
      For an one-class model, +1 or -1 is returned.
      Parameters
      ----------
      X : {array-like, sparse matrix}, shape (n_samples, n_features)
          For kernel="precomputed", the expected shape of X is
          (n_samples_test, n_samples_train).
      Returns
      -------
      y_pred : array, shape (n_samples,)
      """
      X = self._validate_for_predict(X)
      predict = self._sparse_predict if self._sparse else self._dense_predict
      return predict(X)

  def _dense_predict(self, X):
      n_samples, n_features = X.shape
      X = self._compute_kernel(X)
      if X.ndim == 1:
          X = check_array(X, order='C')

      kernel = self.kernel
      if callable(self.kernel):
          kernel = 'precomputed'
          if X.shape[1] != self.shape_fit_[0]:
              raise ValueError("X.shape[1] = %d should be equal to %d, "
                               "the number of samples at training time" %
                               (X.shape[1], self.shape_fit_[0]))

      svm_type = LIBSVM_IMPL.index(self._impl)

      return libsvm.predict(
          X, self.support_, self.support_vectors_, self.n_support_,
          self._dual_coef_, self._intercept_,
          self.probA_, self.probB_, svm_type=svm_type, kernel=kernel,
          degree=self.degree, coef0=self.coef0, gamma=self._gamma,
          cache_size=self.cache_size)

  def _sparse_predict(self, X):
      # Precondition: X is a csr_matrix of dtype np.float64.
      kernel = self.kernel
      if callable(kernel):
          kernel = 'precomputed'

      kernel_type = self._sparse_kernels.index(kernel)

      C = 0.0  # C is not useful here

      return libsvm_sparse.libsvm_sparse_predict(
          X.data, X.indices, X.indptr,
          self.support_vectors_.data,
          self.support_vectors_.indices,
          self.support_vectors_.indptr,
          self._dual_coef_.data, self._intercept_,
          LIBSVM_IMPL.index(self._impl), kernel_type,
          self.degree, self._gamma, self.coef0, self.tol,
          C, self.class_weight_,
          self.nu, self.epsilon, self.shrinking,
          self.probability, self.n_support_,
          self.probA_, self.probB_)

  def _compute_kernel(self, X):
      """Return the data transformed by a callable kernel"""
      if callable(self.kernel):
          # in the case of precomputed kernel given as a function, we
          # have to compute explicitly the kernel matrix
          kernel = self.kernel(X, self.__Xfit)
          if sp.issparse(kernel):
              kernel = kernel.toarray()
          X = np.asarray(kernel, dtype=np.float64, order='C')
      return X

  @deprecated(" and will be removed in 0.19")
  def decision_function(self, X):
      """Distance of the samples X to the separating hyperplane.
      Parameters
      ----------
      X : array-like, shape (n_samples, n_features)
          For kernel="precomputed", the expected shape of X is
          [n_samples_test, n_samples_train].
      Returns
      -------
      X : array-like, shape (n_samples, n_class * (n_class-1) / 2)
          Returns the decision function of the sample for each class
          in the model.
      """
      return self._decision_function(X)

  def _decision_function(self, X):
      """Distance of the samples X to the separating hyperplane.
      Parameters
      ----------
      X : array-like, shape (n_samples, n_features)
      Returns
      -------
      X : array-like, shape (n_samples, n_class * (n_class-1) / 2)
          Returns the decision function of the sample for each class
          in the model.
      """
      # NOTE: _validate_for_predict contains check for is_fitted
      # hence must be placed before any other attributes are used.
      X = self._validate_for_predict(X)
      X = self._compute_kernel(X)

      if self._sparse:
          dec_func = self._sparse_decision_function(X)
      else:
          dec_func = self._dense_decision_function(X)

      # In binary case, we need to flip the sign of coef, intercept and
      # decision function.
      if self._impl in ['c_svc', 'nu_svc'] and len(self.classes_) == 2:
          return -dec_func.ravel()

      return dec_func

  def _dense_decision_function(self, X):
      X = check_array(X, dtype=np.float64, order="C")

      kernel = self.kernel
      if callable(kernel):
          kernel = 'precomputed'

      return libsvm.decision_function(
          X, self.support_, self.support_vectors_, self.n_support_,
          self._dual_coef_, self._intercept_,
          self.probA_, self.probB_,
          svm_type=LIBSVM_IMPL.index(self._impl),
          kernel=kernel, degree=self.degree, cache_size=self.cache_size,
          coef0=self.coef0, gamma=self._gamma)

  def _sparse_decision_function(self, X):
      X.data = np.asarray(X.data, dtype=np.float64, order='C')

      kernel = self.kernel
      if hasattr(kernel, '__call__'):
          kernel = 'precomputed'

      kernel_type = self._sparse_kernels.index(kernel)

      return libsvm_sparse.libsvm_sparse_decision_function(
          X.data, X.indices, X.indptr,
          self.support_vectors_.data,
          self.support_vectors_.indices,
          self.support_vectors_.indptr,
          self._dual_coef_.data, self._intercept_,
          LIBSVM_IMPL.index(self._impl), kernel_type,
          self.degree, self._gamma, self.coef0, self.tol,
          self.C, self.class_weight_,
          self.nu, self.epsilon, self.shrinking,
          self.probability, self.n_support_,
          self.probA_, self.probB_)

  def _validate_for_predict(self, X):
      check_is_fitted(self, 'support_')

      X = check_array(X, accept_sparse='csr', dtype=np.float64, order="C")
      if self._sparse and not sp.isspmatrix(X):
          X = sp.csr_matrix(X)
      if self._sparse:
          X.sort_indices()

      if sp.issparse(X) and not self._sparse and not callable(self.kernel):
          raise ValueError(
              "cannot use sparse input in %r trained on dense data"
              % type(self).__name__)
      n_samples, n_features = X.shape

      if self.kernel == "precomputed":
          if X.shape[1] != self.shape_fit_[0]:
              raise ValueError("X.shape[1] = %d should be equal to %d, "
                               "the number of samples at training time" %
                               (X.shape[1], self.shape_fit_[0]))
      elif n_features != self.shape_fit_[1]:
          raise ValueError("X.shape[1] = %d should be equal to %d, "
                           "the number of features at training time" %
                           (n_features, self.shape_fit_[1]))
      return X

  @property
  def coef_(self):
      if self.kernel != 'linear':
          raise ValueError('coef_ is only available when using a '
                           'linear kernel')

      coef = self._get_coef()

      # coef_ being a read-only property, it's better to mark the value as
      # immutable to avoid hiding potential bugs for the unsuspecting user.
      if sp.issparse(coef):
          # sparse matrix do not have global flags
          coef.data.flags.writeable = False
      else:
          # regular dense array
          coef.flags.writeable = False
      return coef

  def _get_coef(self):
      return safe_sparse_dot(self._dual_coef_, self.support_vectors_)



class BaseSVC(BaseLibSVM, ClassifierMixin):
  """ABC for LibSVM-based classifiers."""

  def _validate_targets(self, y):
      y_ = column_or_1d(y, warn=True)
      cls, y = np.unique(y_, return_inverse=True)
      self.class_weight_ = compute_class_weight(self.class_weight, cls, y_)
      if len(cls) < 2:
          raise ValueError(
              "The number of classes has to be greater than one; got %d"
              % len(cls))

      self.classes_ = cls

      return np.asarray(y, dtype=np.float64, order='C')

  def decision_function(self, X):
      """Distance of the samples X to the separating hyperplane.
      Parameters
      ----------
      X : array-like, shape (n_samples, n_features)
      Returns
      -------
      X : array-like, shape (n_samples, n_class * (n_class-1) / 2)
          Returns the decision function of the sample for each class
          in the model.
      """
      return self._decision_function(X)

  def predict(self, X):
      """Perform classification on samples in X.
      For an one-class model, +1 or -1 is returned.
      Parameters
      ----------
      X : {array-like, sparse matrix}, shape (n_samples, n_features)
          For kernel="precomputed", the expected shape of X is
          [n_samples_test, n_samples_train]
      Returns
      -------
      y_pred : array, shape (n_samples,)
          Class labels for samples in X.
      """
      y = super(BaseSVC, self).predict(X)
      return self.classes_.take(np.asarray(y, dtype=np.intp))

  # Hacky way of getting predict_proba to raise an AttributeError when
  # probability=False using properties. Do not use this in new code; when
  # probabilities are not available depending on a setting, introduce two
  # estimators.
  def _check_proba(self):
      if not self.probability:
          raise AttributeError("predict_proba is not available when"
                               " probability=%r" % self.probability)
      if self._impl not in ('c_svc', 'nu_svc'):
          raise AttributeError("predict_proba only implemented for SVC"
                               " and NuSVC")

  @property
  def predict_proba(self):
      """Compute probabilities of possible outcomes for samples in X.
      The model need to have probability information computed at training
      time: fit with attribute `probability` set to True.
      Parameters
      ----------
      X : array-like, shape (n_samples, n_features)
          For kernel="precomputed", the expected shape of X is
          [n_samples_test, n_samples_train]
      Returns
      -------
      T : array-like, shape (n_samples, n_classes)
          Returns the probability of the sample for each class in
          the model. The columns correspond to the classes in sorted
          order, as they appear in the attribute `classes_`.
      Notes
      -----
      The probability model is created using cross validation, so
      the results can be slightly different than those obtained by
      predict. Also, it will produce meaningless results on very small
      datasets.
      """
      self._check_proba()
      return self._predict_proba

  def _predict_proba(self, X):
      X = self._validate_for_predict(X)
      pred_proba = (self._sparse_predict_proba
                    if self._sparse else self._dense_predict_proba)
      return pred_proba(X)

  @property
  def predict_log_proba(self):
      """Compute log probabilities of possible outcomes for samples in X.
      The model need to have probability information computed at training
      time: fit with attribute `probability` set to True.
      Parameters
      ----------
      X : array-like, shape (n_samples, n_features)
          For kernel="precomputed", the expected shape of X is
          [n_samples_test, n_samples_train]
      Returns
      -------
      T : array-like, shape (n_samples, n_classes)
          Returns the log-probabilities of the sample for each class in
          the model. The columns correspond to the classes in sorted
          order, as they appear in the attribute `classes_`.
      Notes
      -----
      The probability model is created using cross validation, so
      the results can be slightly different than those obtained by
      predict. Also, it will produce meaningless results on very small
      datasets.
      """
      self._check_proba()
      return self._predict_log_proba

  def _predict_log_proba(self, X):
      return np.log(self.predict_proba(X))

  def _dense_predict_proba(self, X):
      X = self._compute_kernel(X)

      kernel = self.kernel
      if callable(kernel):
          kernel = 'precomputed'

      svm_type = LIBSVM_IMPL.index(self._impl)
      pprob = libsvm.predict_proba(
          X, self.support_, self.support_vectors_, self.n_support_,
          self._dual_coef_, self._intercept_,
          self.probA_, self.probB_,
          svm_type=svm_type, kernel=kernel, degree=self.degree,
          cache_size=self.cache_size, coef0=self.coef0, gamma=self._gamma)

      return pprob

  def _sparse_predict_proba(self, X):
      X.data = np.asarray(X.data, dtype=np.float64, order='C')

      kernel = self.kernel
      if callable(kernel):
          kernel = 'precomputed'

      kernel_type = self._sparse_kernels.index(kernel)

      return libsvm_sparse.libsvm_sparse_predict_proba(
          X.data, X.indices, X.indptr,
          self.support_vectors_.data,
          self.support_vectors_.indices,
          self.support_vectors_.indptr,
          self._dual_coef_.data, self._intercept_,
          LIBSVM_IMPL.index(self._impl), kernel_type,
          self.degree, self._gamma, self.coef0, self.tol,
          self.C, self.class_weight_,
          self.nu, self.epsilon, self.shrinking,
          self.probability, self.n_support_,
          self.probA_, self.probB_)

  def _get_coef(self):
      if self.dual_coef_.shape[0] == 1:
          # binary classifier
          coef = safe_sparse_dot(self.dual_coef_, self.support_vectors_)
      else:
          # 1vs1 classifier
          coef = _one_vs_one_coef(self.dual_coef_, self.n_support_,
                                  self.support_vectors_)
          if sp.issparse(coef[0]):
              coef = sp.vstack(coef).tocsr()
          else:
              coef = np.vstack(coef)

      return coef


class SVC(BaseSVC):
  """C-Support Vector Classification.
  The implementation is based on libsvm. The fit time complexity
  is more than quadratic with the number of samples which makes it hard
  to scale to dataset with more than a couple of 10000 samples.
  The multiclass support is handled according to a one-vs-one scheme.
  For details on the precise mathematical formulation of the provided
  kernel functions and how `gamma`, `coef0` and `degree` affect each
  other, see the corresponding section in the narrative documentation:
  :ref:`svm_kernels`.
  .. The narrative documentation is available at http://scikit-learn.org/
  Parameters
  ----------
  C : float, optional (default=1.0)
      Penalty parameter C of the error term.
  kernel : string, optional (default='rbf')
       Specifies the kernel type to be used in the algorithm.
       It must be one of 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' or
       a callable.
       If none is given, 'rbf' will be used. If a callable is given it is
       used to precompute the kernel matrix.
  degree : int, optional (default=3)
      Degree of the polynomial kernel function ('poly').
      Ignored by all other kernels.
  gamma : float, optional (default=0.0)
      Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
      If gamma is 0.0 then 1/n_features will be used instead.
  coef0 : float, optional (default=0.0)
      Independent term in kernel function.
      It is only significant in 'poly' and 'sigmoid'.
  probability : boolean, optional (default=False)
      Whether to enable probability estimates. This must be enabled prior
      to calling `fit`, and will slow down that method.
  shrinking : boolean, optional (default=True)
      Whether to use the shrinking heuristic.
  tol : float, optional (default=1e-3)
      Tolerance for stopping criterion.
  cache_size : float, optional
      Specify the size of the kernel cache (in MB).
  class_weight : {dict, 'auto'}, optional
      Set the parameter C of class i to class_weight[i]*C for
      SVC. If not given, all classes are supposed to have
      weight one. The 'auto' mode uses the values of y to
      automatically adjust weights inversely proportional to
      class frequencies.
  verbose : bool, default: False
      Enable verbose output. Note that this setting takes advantage of a
      per-process runtime setting in libsvm that, if enabled, may not work
      properly in a multithreaded context.
  max_iter : int, optional (default=-1)
      Hard limit on iterations within solver, or -1 for no limit.
  random_state : int seed, RandomState instance, or None (default)
      The seed of the pseudo random number generator to use when
      shuffling the data for probability estimation.
  Attributes
  ----------
  support_ : array-like, shape = [n_SV]
      Indices of support vectors.
  support_vectors_ : array-like, shape = [n_SV, n_features]
      Support vectors.
  n_support_ : array-like, dtype=int32, shape = [n_class]
      Number of support vectors for each class.
  dual_coef_ : array, shape = [n_class-1, n_SV]
      Coefficients of the support vector in the decision function. \
      For multiclass, coefficient for all 1-vs-1 classifiers. \
      The layout of the coefficients in the multiclass case is somewhat \
      non-trivial. See the section about multi-class classification in the \
      SVM section of the User Guide for details.
  coef_ : array, shape = [n_class-1, n_features]
      Weights assigned to the features (coefficients in the primal
      problem). This is only available in the case of linear kernel.
      `coef_` is a readonly property derived from `dual_coef_` and
      `support_vectors_`.
  intercept_ : array, shape = [n_class * (n_class-1) / 2]
      Constants in decision function.
  Examples
  --------
  >>> import numpy as np
  >>> X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
  >>> y = np.array([1, 1, 2, 2])
  >>> from sklearn.svm import SVC
  >>> clf = SVC()
  >>> clf.fit(X, y) #doctest: +NORMALIZE_WHITESPACE
  SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,
      gamma=0.0, kernel='rbf', max_iter=-1, probability=False,
      random_state=None, shrinking=True, tol=0.001, verbose=False)
  >>> print(clf.predict([[-0.8, -1]]))

  """
  def __init__(self, C=1.0, kernel='rbf', degree=3, gamma=0.0,
               coef0=0.0, shrinking=True, probability=False,
               tol=1e-3, cache_size=200, class_weight=None,
               verbose=False, max_iter=-1, random_state=None):


from sklearn import datasets, svm
import pylab as pl

digits = datasets.load_digits()
pl.imshow(digits.images[0], cmap=pl.cm.gray_r)
data = digits.images.reshape((digits.images.shape[0], -1))

clf = svm.LinearSVC()
# learn from the data
clf.fit(iris.data, iris.target)
clf.predict([[ 5.0,  3.6,  1.3,  0.25]])
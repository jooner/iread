import cPickle, gzip, numpy

# 1. first step : normalize image data from NMIST

# Load the dataset
f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()
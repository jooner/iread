import cPickle, gzip, numpy

# 1. first step : input data & normalize image data from NMIST

# Load the dataset
f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

# 2. second step : randomly place our k centroids in random locations

# 				LOOP STARTS 
# 3. third step : classify each data point into one of our k clusters
# 	 depending on which centroid it's closest to  (use Euclidan distance)


# 4. fourth step : the location of each centroid should change to be 
#    the average of all the location values of its data points

# 5. fifth step ; classify each data point using NEW k location, repeat loop etc.

# 6. sixth step : when the ^ doesn't change the location values, then THE LOOP STOPS

# 7. seventh step : you have your clusters  
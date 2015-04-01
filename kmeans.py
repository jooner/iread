
import cPickle, gzip, numpy

# 1. first step : input data & normalize image data from NMIST

# Load the dataset

# not using scikit
# f = gzip.open('mnist.pkl.gz', 'rb')
# train_set, valid_set, test_set = cPickle.load(f)
# f.close()

# using scikit 
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original', data_home=custom_data_home)


# 2. second step : randomly place our k centroids in random locations



# 				LOOP STARTS 
# 3. third step : classify each data point into one of our k clusters
# 	 depending on which centroid it's closest to  (use Euclidan distance)

# a. find the nearest cluster center
def nearest_cluster_center(point, cluster_centers):
    def sqr_distance_2D(a, b):
        return (a.x - b.x) ** 2  +  (a.y - b.y) ** 2
 
 


# 4. fourth step : the location of each centroid should change to be 
#    the average of all the location values of its data points

# 5. fifth step ; classify each data point using NEW k location, repeat loop etc.

# 6. sixth step : when the ^ doesn't change the location values, then THE LOOP STOPS

# 7. seventh step : you have your clusters  
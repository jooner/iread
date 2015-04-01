
# coding: utf-8

# In[66]:

get_ipython().magic(u'matplotlib inline')
import os
import struct

import cPickle
import gzip
import tempfile
import urllib
import numpy as nmpy
import matplotlib.pyplot as matplt
from __future__ import division
import scipy

#nice defaults for matplotlib
from matplotlib import rcParams

dark2_colors = [(0.10588235294117647, 0.6196078431372549, 0.4666666666666667),
                (0.8509803921568627, 0.37254901960784315, 0.00784313725490196),
                (0.4588235294117647, 0.4392156862745098, 0.7019607843137254),
                (0.9058823529411765, 0.1607843137254902, 0.5411764705882353),
                (0.4, 0.6509803921568628, 0.11764705882352941),
                (0.9019607843137255, 0.6705882352941176, 0.00784313725490196),
                (0.6509803921568628, 0.4627450980392157, 0.11372549019607843),
                (0.4, 0.4, 0.4)]

rcParams['figure.figsize'] = (10, 6)
rcParams['figure.dpi'] = 150
rcParams['axes.color_cycle'] = dark2_colors
rcParams['lines.linewidth'] = 2
rcParams['axes.grid'] = True
rcParams['axes.facecolor'] = '#eeeeee'
rcParams['font.size'] = 14
rcParams['patch.edgecolor'] = 'none'



let datalength = 60000
# ## K-means implementation

# In[77]:

# import the training images dataset 
# make sure the extracted file is in the same folder as the iPython notebook
trainingdata = nmpy.fromfile('train-images.idx3-ubyte', dtype='int8' , sep="")


# In[78]:

# get rid of the header in the dataset
trainingdata = trainingdata[16:]


# In[79]:

# reshape the dataset
digit = trainingdata.reshape((datalength,28,28))


# In[80]:

# ensure positive values
digit = digit%256


# In[81]:

# plot first digit in dataset
matplt.imshow(digit[0], cmap='afmhot')
matplt.show()


# In[11]:

# create a 784 dimension vector from trainingdata
trainingdata = trainingdata.reshape((datalength, 784))
trainingdata = trainingdata%256


# In[12]:

# sanity check
matplt.imshow(trainingdata[0].reshape((28,28)), cmap='afmhot')
matplt.show()


# In[53]:

# plotting functions
def plotofmeans(avg):
    
    # number of clusters (our k will be 10 for each digit)
    K = len(avg)
    num_cols = int(nmpy.ceil(nmpy.sqrt(K)))
    num_rows = nmpy.ceil(K/num_cols)
    
    rcParams['figure.figsize'] = (5, 5*num_rows/num_cols)
    fig = matplt.gcf()
    matplt.figure(1)
    
    for k in range(K):
        matplt.subplot(num_rows, num_cols, k+1)
        frame = matplt.gca()
        matplt.imshow(avg[k].reshape((28,28)), cmap='afmhot')
        
    matplt.show()

def plot_examples(K, var, n):
    for k in range(K):
        individual = nmpy.where(var==k)[0]
        plotofmeans(trainingdata[nmpy.random.choice(individual, replace = False, size = n)])
        
def plot_objf(km_dict):
    rcParams['figure.figsize'] = (10, 6)
    matplt.plot(range(len(km_dict['J'])),km_dict['J'])
    matplt.xlim(0, len(km_dict['J']))
    matplt.ylabel("J (distortion measure)")
    matplt.xlabel("Iteration")
    matplt.show()


# In[39]:

# initialize each datapoint with a random cluster initialization
def init_each_k(K):
    return nmpy.array([nmpy.random.randint(0,K) for i in range(datalength)])


# In[54]:

def k_means(K, init = init_each_k):
    
    # dictionary of responsibility vectors, means, and distortion
    d = {}
    
    #variables
    avgs  = nmpy.zeros((K,784))
    Distort = []
    regenerate = 1
    
    #initial center assignment 
    center = init(K)
    init_in_dict = False
    
    while regenerate:
        
        # compute means of each cluster using numpy 
        avgs = nmpy.array([nmpy.mean(trainingdata[nmpy.where(center == k)[0]], axis = 0) for k in range(K)])
        
        # add initial means to dictionary
        if not init_in_dict:
            d['init'] = avgs
            init_in_dict = True
        
        # regenerate clusters by calculating the distance from each cluster mean
        # regenerate the responsibility to the smallest distance
        distance = [nmpy.linalg.norm(trainingdata-avgs[k], axis = 1) for k in range(K)]
        new_center = nmpy.argmin(distance, axis = None)
        
        # the number of datapoints that were reassigned in each iteration
        regenerate = nmpy.count_nonzero(center-new_center)
        center = new_center
        
        # distortion measure: measure of error
        # Distort should be decreasing with each iteration
        Distort.append(nmpy.sum(nmpy.min(nmpy.array(distance)**2, axis = 0)))
        print regenerate, "regenerated Distort:", Distort[-1]/1e06
       
    # add the responsibilities, means, and distortion to the dictionary
    d.update({"center":center, "means":avgs, "J":Distort})
    return d
        


# In[41]:

# run k_means with 5 clusters
# be patient while it runs until 0 reassignments
run5 = k_means(5)


# In[52]:

# the means of our 5 clusters
plotofmeans(run5['means'])


# In[27]:

plot_examples(5, run5['center'], 25)


# In[36]:

# distortion measure J plotted against iterations
plot_objf(run5)


# In[23]:

# run k_means with 10 clusters -- one for each digit
run10 = k_means(10)


# In[ ]:

plotofmeans(run10['means'])


# In[ ]:

plot_examples(5, run10['center'], 25)


# In[ ]:

# distortion measure plotted against iterations
plot_objf(run10)


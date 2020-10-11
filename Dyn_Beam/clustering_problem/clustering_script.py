import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans

import pdb

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

data1 = np.loadtxt('./simulations/s1_cantilever_EIz_1.txt') 
data2 = np.loadtxt('./simulations/s2_cantilever_EIz_4.txt') 


#Clean data

data1 = np.delete(data1, obj = range(10), axis=0)
data2 = np.delete(data2, obj = range(10), axis=0)

#Sensors located at x = [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
#Using info from [.3, .6, .9]


X_train = np.array([data1[:, 2], data1[:, 5], data1[:, 8]])

X_train = np.concatenate((X_train, [data2[:, 2], data2[:, 5], data2[:, 8]]), axis = 1)
                    


kmeans = KMeans(n_clusters=2, init = 'k-means++', max_iter=1000, n_init = 100, random_state=0)
kmeans.fit(X_train.transpose())

#post-process


labels = kmeans.labels_
print(kmeans.cluster_centers_)
print(kmeans.labels_)

X_train = X_train.transpose()
ax = plt.axes(projection='3d')
ax.scatter(X_train[labels==0,0], X_train[labels==0,1], X_train[labels==0,2], c='r', marker='o');
ax.scatter(X_train[labels==1,0], X_train[labels==1,1], X_train[labels==1,2], c='b', marker='x');
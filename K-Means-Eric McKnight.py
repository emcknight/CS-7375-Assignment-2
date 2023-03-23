# Required imports for K-Means Implementation
import matplotlib.pyplot as plt
import numpy as np
import math
import seaborn as sns
from matplotlib import style

# Import the toy dataset
X = np.array([[2, 4],
 [1.7, 2.8],
 [7, 8],
 [8.6, 8],
 [3.4, 1.5],
 [9,11]])
plt.scatter(X[:,0], X[:,1], s=150)
plt.show()

# Euclidean Distance Function for a point with any amount of variables. The two points must have the same number of variables.
def euclidean(a, b):
  dist = 0  # Sum variable

  # For each variable in the x point...
  for i in range(len(a)):
    tmp = (b[i]-a[i]) ** 2  # Calculated the squared difference between x and y's given variable in the for loop
    dist += tmp  # Add the squared difference to the sum variable
  eucDist = math.sqrt(dist)  # Take the square root of the sum to finalize the euclidean distance calculation.
  return eucDist

# Function to find the closest centroid for a given point.
def closestCentroid(data, centroids):
  assignments = []
  for point in data:  # for each point...
    distance = []
    for centroid in centroids:  # check each centroid's distance to the point
      distance.append(euclidean(point, centroid))
    assignments.append(np.argmin(distance))  # Take the minimum distance as the centroid for assignment
  return np.array(assignments)  # Return a list of assignments for each point

# Function to calculate the average midpoint for each cluster assignment.
def updateCentroid(data, clusters, k):
  new_centroids = []
  for c in range(k):  # For each centroid...
    cluster_mean = data[clusters == c].mean(axis=0)  # Find the average of the columns of each cluster (x & y in this case)
    new_centroids.append(cluster_mean) # Create a new list of centroids
  return np.array(new_centroids)  # Return the new list of centroids

# K-Means driver function
def kmeans(data, k, i):
  centroids = []
  num_dims = np.shape(data)[1]  # Calculates the number of dimensions the data has
  min = np.amin(data)  # Find the minimum value of all data points in the toy dataset
  max = np.amax(data)  # Find the maximum value of all data points in the toy dataset
  centroids = np.random.uniform(low=min, high=max, size=(k,num_dims))  # Create random centroids based on the data passed
  print('Initialized centroids:\n {}'.format(centroids))
  for i in range(i):
    clusters = closestCentroid(data, centroids)  # Create centroid assignments for each of the datapoints
    centroids = updateCentroid(data, clusters, k)  # Update the centroids based on the assignments
    print('Iteration: {}, Centroids:\n {}'.format(1, centroids))
  return centroids, clusters

# Driver code to initiate the function
k = 2  # The number of clusters to be created
i = 10  # The number of iterations the the K-Means algorithm should run
centroids, clusters = kmeans(X, k, i)

# Visualization of clusters using Seaborn/MatPlotLib
# Scatterplot to show the points and their groupings of cluster assignments by color
sns.scatterplot(x=[X[0] for X in X],
                y=[X[1] for X in X],
                hue=clusters,
                legend=None
                )

# Plotting of the Centroids by using an 'x'
plt.plot([x for x, _ in centroids],
         [y for _, y in centroids],
         'x',
         markersize=10)
plt.show()

# K-Means Clustering

# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset with pandas

dataset = pd.read_csv("Mall_Customers.csv")
X = dataset.iloc[:, [3,4]].values
print("-------------Array of Values-----------")
print(X)

# Using the Elbow Method to find the optimum number of clusters
from sklearn.cluster import KMeans

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
 
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Applying K means to the dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
Y_kmeans = kmeans.fit_predict(X)
print("--------------Predictions of Clusters-------------")
print(Y_kmeans)


# Visualising the Clusters
plt.scatter(X[Y_kmeans==0, 0], X[Y_kmeans == 0, 1], s = 100, c = 'red', label = 'Standard Clients' )

plt.scatter(X[Y_kmeans==1, 0], X[Y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Careless Clients' )

plt.scatter(X[Y_kmeans==2, 0], X[Y_kmeans == 2, 1], s = 100, c = 'green', label = 'Target Clients' )

plt.scatter(X[Y_kmeans == 3, 0], X[Y_kmeans == 3, 1], s=100, c='cyan', label='Sensible Clients')

plt.scatter(X[Y_kmeans == 4, 0], X[Y_kmeans == 4, 1], s=100, c='magenta', label='Careful Clients')

# Plotting the centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')

plt.title('Clusters of Clients')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()




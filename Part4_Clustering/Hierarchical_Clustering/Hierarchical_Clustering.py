# Hierarchical Clustering

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import the dataset with pandas
dataset = pd.read_csv("Mall_Customers.csv")
X = dataset.iloc[:, [3, 4]].values
print("----------Array of Values-------------")
print(X)

# Using the Dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distances')
plt.show()

# Fitting Hierarchical clustering to the dataset 
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
Y_hc = hc.fit_predict(X)
print("-----------New Vector of predicted clusters-------------")
print(Y_hc)

# Visualising the Clusters 
plt.scatter(X[Y_hc == 0, 0], X[Y_hc == 0, 1],
            s=100, c='red', label='Careful Customers')

plt.scatter(X[Y_hc == 1, 0], X[Y_hc == 1, 1],
            s=100, c='blue', label='Standard Customers')

plt.scatter(X[Y_hc == 2, 0], X[Y_hc == 2, 1],
            s=100, c='green', label='Target Customers')

plt.scatter(X[Y_hc == 3, 0], X[Y_hc == 3, 1],
            s=100, c='cyan', label='Careless Customers')

plt.scatter(X[Y_hc == 4, 0], X[Y_hc == 4, 1],
            s=100, c='magenta', label='Sensible Customers')


plt.title('Clusters of Clients')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

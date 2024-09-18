import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

# Example data
x = np.array([
    [1, 2], [2, 3], [3, 4], [5, 6], [8, 8],
    [7, 5], [9, 7], [6, 6], [4, 2], [2, 1],
    [0, 0], [10, 10], [12, 12], [3, 7], [6, 3]
])

# Perform agglomerative clustering using sklearn
agg_clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0, linkage='complete')
agg_clustering.fit(x)

# Generate the linkage matrix using scipy
Z = linkage(x, method='complete')

print(Z)

# Plot the dendrogram
plt.figure(figsize=(10, 7))
plt.title("Dendrogram for Complete Linkage Clustering")
dendrogram(Z)
plt.show()

# Print final cluster labels
print("Final cluster labels:", agg_clustering.labels_)

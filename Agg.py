import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

def EuclideanDistance(a, b):
    return np.sqrt(np.sum((a - b)**2))

def DistanceMatrix(x):
    n = np.shape(x)[0]
    distanceMatrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            distance = EuclideanDistance(x[i], x[j])
            distanceMatrix[i, j] = distance
            distanceMatrix[j, i] = distance
    return distanceMatrix

def CompleteLinking(distanceMatrix, clusters):
    min_max_dist = np.inf
    cluster_pair = (-1, -1)

    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            max_dist = -np.inf
            for point_i in clusters[i]:
                for point_j in clusters[j]:
                    dist = distanceMatrix[point_i, point_j]
                    if dist > max_dist:
                        max_dist = dist
            if max_dist < min_max_dist:
                min_max_dist = max_dist
                cluster_pair = (i, j)

    return cluster_pair

def AgglomerativeClustering(x):
    n = np.shape(x)[0]
    clusters = [[i] for i in range(n)]
    distanceMatrix = DistanceMatrix(x)
    print("Initial Distance Matrix:")
    print(distanceMatrix)

    while len(clusters) > 1:
        mergeIndex = CompleteLinking(distanceMatrix, clusters)
        clusters[mergeIndex[0]].extend(clusters[mergeIndex[1]])
        del clusters[mergeIndex[1]]
        print("Merged clusters:", mergeIndex)
        print("Current clusters:", clusters)

    return clusters

# Example usage
x = np.array([
    [1, 2], [2, 3], [3, 4], [5, 6], [8, 8],
    [7, 5], [9, 7], [6, 6], [4, 2], [2, 1],
    [0, 0], [10, 10], [12, 12], [3, 7], [6, 3]
])
clusters = AgglomerativeClustering(x)
print("Final clusters:", clusters)

# Plotting dendrogram
Z = linkage(x, method='complete')

plt.figure(figsize=(10, 7))
plt.title("Dendrogram for Complete Linkage Clustering")
dendrogram(Z)
plt.show()
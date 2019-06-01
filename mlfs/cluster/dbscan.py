import numpy as np
import random


def _construct_neigbors(X, eps):
    n_samples = X.shape[0]
    neighbors = [[] for _ in range(n_samples)]
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            dist = np.linalg.norm(X[i]-X[j])
            if dist < eps:
                neighbors[i].append(j)
                neighbors[j].append(i)
    return neighbors


def _construct_cores(neighbors, min_samples):
    cores = set()
    for i, neighbor in enumerate(neighbors):
        if len(neighbor) >= min_samples:
            cores.add(i)
    return cores


def _bfs(start, neighbors, visited, min_samples):
    """Use bfs to search all density-reachable samples from start"""
    queue = [start]
    visited[start] = 1
    cluster = set([start])
    while queue:
        point = queue.pop(0)
        if len(neighbors[point]) >= min_samples:
            for neighbor in neighbors[point]:
                if not visited[neighbor]:
                    queue.append(neighbor)
                    visited[neighbor] = 1
                    cluster.add(neighbor)
    return cluster


def dbscan(X, eps=0.5, min_samples=5):
    """Perform DBSCAN clustering."""
    neighbors = _construct_neigbors(X, eps)
    cores = _construct_cores(neighbors, min_samples)

    n_samples = X.shape[0]
    labels = -np.ones(n_samples, dtype=int)
    visited = np.zeros(n_samples, dtype=int)
    n_cluster = 0

    while cores:
        start = random.choice(list(cores))
        cluster = _bfs(start, neighbors, visited, min_samples)
        for i in cluster:
            labels[i] = n_cluster
        cores -= cluster
        n_cluster += 1

    return labels


class DBSCAN(object):
    """Perform DBSCAN clustering

    DBSCAN - Density-Based Spatial Clustering of Applications with Noise.
    Finds core samples of high density and expands clusters from them.
    Good for data which contains clusters of similar density.

    Parameters
    ----------
    eps : float, optional
        The maximum distance between two samples for one to be considered
        as in the neighborhood of the other. This is not a maximum bound
        on the distances of points within a cluster. This is the most
        important DBSCAN parameter to choose appropriately for your data set
        and distance function.
    min_samples : int, optional
        The number of samples (or total weight) in a neighborhood for a point
        to be considered as a core point. This includes the point itself.
    """

    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples

    def fit(self, X):
        """Perform DBSCAN clustering from features."""
        self.labels_ = dbscan(X, eps=self.eps, min_samples=self.min_samples)
        return self

    def fit_predict(self, X):
        """Performs clustering on X and returns cluster labels."""
        return self.fit(X).labels_

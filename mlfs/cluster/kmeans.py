import numpy as np


def _init_random_centroids(X, n_clusters):
    """Randomly init n_clusters centroids"""
    return X[np.random.randint(0, X.shape[0], n_clusters)]


def _closest_centroid(sample, centroids):
    """Find the index of the centroid that is closest to sample"""
    dist = np.sum(np.square(centroids - sample), axis=1)
    return np.argmin(dist)


def _create_clusters(centroids, X):
    """Cluster samples to centroids"""
    clusters = [[] for i in range(len(centroids))]
    for i in range(X.shape[0]):
        centroid_index = _closest_centroid(X[i], centroids)
        clusters[centroid_index].append(i)
    return clusters


def _update_centroids(clusters, X):
    """Update centroids."""
    new_centroids = np.zeros((len(clusters), X.shape[1]))
    for i, cluster in enumerate(clusters):
        new_centroids[i, :] = np.mean(X[cluster], axis=0)
    return new_centroids


def _get_cluster_labels(centroids, X):
    n_samples = X.shape[0]
    labels = np.zeros(n_samples)
    for i in range(n_samples):
        labels[i] = _closest_centroid(X[i], centroids)
    return labels


def k_means(X, n_clusters=3, max_iter=500, tol=1e-4):
    """Compute k-means clustering and return n_clusters centroids."""
    centroids = _init_random_centroids(X, n_clusters)
    for _ in range(max_iter):
        clusters = _create_clusters(centroids, X)
        new_centroids = _update_centroids(clusters, X)
        # If centroids have almost no changes, we consider it having converged.
        diff = np.sum(np.abs(new_centroids - centroids), axis=1)
        if np.all(diff < tol):
            break
        centroids = new_centroids
    labels = _get_cluster_labels(centroids, X)
    return centroids, labels


class KMeans(object):
    """K-Means clustering

    Parameters:
    -----------
    n_clusters : int, optional, default: 3
        The number of clusters to form as well as the number of
        centroids to generate.

    max_iter : int, optional, default: 500
        Maximum number of iterations of the k-means algorithm to run.

    tol : float, optional, default: 1e-4
        The relative increment in the results before declaring convergence.
    """

    def __init__(self, n_clusters=3, max_iter=500, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        """Compute k-means clustering."""
        self.cluster_centers_, self.labels_ = k_means(
            X, n_clusters=self.n_clusters, max_iter=self.max_iter, tol=self.tol)
        return self

    def predict(self, X):
        """Predict the closest cluster each sample in X belongs to."""
        return _get_cluster_labels(self.cluster_centers_, X)

    def fit_predict(self, X):
        """Compute cluster centers and predict cluster index for each sample."""
        return self.fit(X).labels_

import numpy as np


def dist_min(X, cluster1, cluster2):
    min_dist = 1e12
    for i in cluster1:
        for j in cluster2:
            dist = np.linalg.norm(X[i] - X[j])
            if dist < min_dist:
                min_dist = dist
    return min_dist


def dist_max(X, cluster1, cluster2):
    max_dist = 0
    for i in cluster1:
        for j in cluster2:
            dist = np.linalg.norm(X[i] - X[j])
            if dist > max_dist:
                max_dist = dist
    return max_dist


def dist_avg(X, cluster1, cluster2):
    sum_dist = 0
    for i in cluster1:
        for j in cluster2:
            sum_dist += np.linalg.norm(X[i] - X[j])
    avg_dist = sum_dist / (len(cluster1) * len(cluster2))
    return avg_dist


class AgglomerativeClustering(object):
    """
    Agglomerative Clustering

    Recursively merges the pair of clusters that minimally increases
    a given linkage distance.

    Parameters
    ----------
    n_clusters : int or None, optional, default: 2
        The number of clusters to find.

    linkage : {"complete", "average", "single"}, optional, default: 'complete'
        Which linkage criterion to use. The linkage criterion determines which
        distance to use between sets of observation. The algorithm will merge
        the pairs of cluster that minimize this criterion.

        - average uses the average of the distances of each observation of
          the two sets.
        - complete or maximum linkage uses the maximum distances between
          all observations of the two sets.
        - single uses the minimum of the distances between all observations
          of the two sets.
    """

    def __init__(self, n_clusters=2, linkage='complete'):
        self.n_clusters = n_clusters
        self.linkage = linkage

    def fit(self, X):
        clusters = [set([i]) for i in range(X.shape[0])]
        while len(clusters) > self.n_clusters:
            i, j = self._find_closest_clusters(clusters, X)
            clusters[i] |= clusters.pop(j)

        self.labels_ = -np.ones(X.shape[0], dtype=int)
        for n, cluster in enumerate(clusters):
            for i in cluster:
                self.labels_[i] = n
        return self

    def fit_predict(self, X):
        return self.fit(X).labels_

    def _find_closest_clusters(self, clusters, X):
        min_dist = 1e12
        closest_i, closest_j = 0, 0
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                dist = self._dist(X, clusters[i], clusters[j])
                if dist < min_dist:
                    min_dist = dist
                    closest_i, closest_j = i, j
        return closest_i, closest_j

    def _dist(self, X, cluster1, cluster2):
        if self.linkage == 'average':
            return dist_avg(X, cluster1, cluster2)
        if self.linkage == 'complete':
            return dist_max(X, cluster1, cluster2)
        if self.linkage == 'single':
            return dist_min(X, cluster1, cluster2)
        raise ValueError('invalid linkage: %s' % self.linkage)

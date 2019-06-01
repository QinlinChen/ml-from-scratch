import numpy as np
from scipy.stats import multivariate_normal


class GaussianMixture(object):
    """Gaussian Mixture.

    Representation of a Gaussian mixture model probability distribution.
    This class allows to estimate the parameters of a Gaussian mixture
    distribution.

    Parameters
    ----------
    n_components : int, defaults to 1.
        The number of mixture components.

    max_iter : int, defaults to 100.
        The number of EM iterations to perform.
    """

    def __init__(self, n_components=1, max_iter=100):
        self.n_components = n_components
        self.max_iter = max_iter

    def fit(self, X):
        """Estimate model parameters with the EM algorithm."""
        row, col = X.shape
        self.weights_ = np.ones(self.n_components) / self.n_components
        self.means_ = X[np.random.randint(0, row, self.n_components)]
        self.covariances_ = np.array([np.eye(col) / 10] * self.n_components)

        for _ in range(self.max_iter):
            post_prob = self._calc_post_prob(X)
            self._update_param(post_prob, X)

        return self

    def _calc_post_prob(self, X):
        n_samples = X.shape[0]
        prob_mat = np.zeros((n_samples, self.n_components))
        for i in range(n_samples):
            for j in range(self.n_components):
                prob_mat[i, j] = multivariate_normal.pdf(
                    X[i], mean=self.means_[j], cov=self.covariances_[j])
        prob_mat *= self.weights_
        prob_mat /= np.sum(prob_mat, axis=1, keepdims=True)
        return prob_mat

    def _update_param(self, post_prob, X):
        for i in range(self.n_components):
            gamma = post_prob[:, i]
            self.means_[i] = np.average(X, axis=0, weights=gamma)
            centered_data = X - self.means_[i]
            self.covariances_[i] = np.dot(
                centered_data.T * gamma, centered_data)
            self.covariances_[i] /= np.sum(gamma)
        self.weights_ = np.mean(post_prob, axis=0)

    def predict(self, X):
        """Predict the labels for the data samples in X using trained model."""
        post_prob = self._calc_post_prob(X)
        pred = np.argmax(post_prob, axis=1)
        return pred

    def predict_proba(self, X):
        """Predict posterior probability of each component given the data."""
        return self._calc_post_prob(X)

    def fit_predict(self, X):
        """Estimate model parameters using X and predict the labels for X."""
        return self.fit(X).predict(X)

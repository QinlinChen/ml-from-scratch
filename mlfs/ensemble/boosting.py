import numpy as np
from sklearn.tree import DecisionTreeClassifier


def sample(X, y, distrib):
    nsample = len(distrib)
    index = np.random.choice(nsample, nsample, p=distrib)
    return X[index], y[index]


class AdaBoostClassifier(object):
    """An AdaBoost classifier.

    An AdaBoost classifier is a meta-estimator that begins by fitting a
    classifier on the original dataset and then fits additional copies of the
    classifier on the same dataset but where the weights of incorrectly
    classified instances are adjusted such that subsequent classifiers focus
    more on difficult cases.

    Parameters
    ----------
    n_estimators : integer, optional (default=50)
        The maximum number of estimators at which boosting is terminated.
        In case of perfect fit, the learning procedure is stopped early.
    """

    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators if n_estimators > 0 else 1

    def fit(self, X, y):
        """Build a boosted classifier from the training set (X, y)."""
        self.estimators_ = [DecisionTreeClassifier(max_depth=1)
                            for i in range(self.n_estimators)]
        self.alpha = np.zeros(self.n_estimators)
        nsample = X.shape[0]
        distrib = np.ones(nsample) / nsample

        for i in range(self.n_estimators):
            err = 1.0
            while err > 0.5:
                self.estimators_[i].fit(*sample(X, y, distrib))
                predicted = self.estimators_[i].predict(X)
                err = np.sum((predicted != y) * distrib)
            self.alpha[i] = np.log((1 - err) / np.max([err, 1e-8])) / 2
            distrib *= np.exp(self.alpha[i] * (1 - 2 * (predicted == y)))
            distrib /= distrib.sum()
        return self

    def predict(self, X):
        """Predict classes for X."""
        pred = np.zeros(X.shape[0])
        for i in range(self.n_estimators):
            pred += self.alpha[i] * (2 * self.estimators_[i].predict(X) - 1)
        pred = np.sign(pred)
        pred[pred < 0] = 0
        return pred

    def predict_proba(self, X):
        """Predict class probabilities for X."""
        prob_mat = np.zeros((X.shape[0], self.estimators_[0].n_classes_))
        for i in range(self.n_estimators):
            prob_mat += self.alpha[i] * self.estimators_[i].predict_proba(X)
        prob_mat /= np.sum(self.alpha)
        return prob_mat

import numpy as np
from sklearn.tree import DecisionTreeClassifier


def bootstrap(X, y):
    nsample = X.shape[0]
    index = np.random.randint(0, nsample, nsample)
    return X[index], y[index]


class RandomForestClassifier(object):
    """A random forest classifier.

    A random forest is a meta estimator that fits a number of decision tree
    classifiers on various sub-samples of the dataset and uses averaging to
    improve the predictive accuracy and control over-fitting.
    The sub-sample size is always the same as the original input sample size.

    Parameters
    ----------
    n_estimators : integer, optional (default=100)
        The number of trees in the forest.
    """

    def __init__(self, n_estimators=100):
        self.n_estimators = n_estimators if n_estimators > 0 else 1

    def fit(self, X, y):
        self.estimators_ = [DecisionTreeClassifier(max_features='log2')
                            for i in range(self.n_estimators)]
        for i in range(self.n_estimators):
            self.estimators_[i].fit(*bootstrap(X, y))
        return self

    def predict(self, X):
        pred_mat = np.zeros((X.shape[0], self.n_estimators))
        for i in range(self.n_estimators):
            pred_mat[:, i] = self.estimators_[i].predict(X)
        pred = np.zeros(X.shape[0])
        for i in range(len(pred)):
            classes, counts = np.unique(pred_mat[i], return_counts=True)
            pred[i] = classes[np.argmax(counts)]
        return pred

    def predict_proba(self, X):
        prob_mat = np.zeros((X.shape[0], self.estimators_[0].n_classes_))
        for i in range(self.n_estimators):
            prob_mat += self.estimators_[i].predict_proba(X)
        prob_mat /= self.n_estimators
        return prob_mat

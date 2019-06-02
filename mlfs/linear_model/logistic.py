import numpy as np
from scipy import linalg


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _log_likelihood(X, y, w):
    z = X.dot(w)
    return np.sum(np.log(1 + np.exp(z)) - z * y)


def _binarize_label(y, pos_label):
    """Set pos_label in y to 1.0 and others to 0.0."""
    norm_y = y.copy()
    mask = (norm_y == pos_label)
    norm_y[mask] = 1.0
    norm_y[np.logical_not(mask)] = 0.0
    return norm_y


def _hstack_ones_to_data(X):
    """Append ones to the last column of X."""
    n = X.shape[0]
    return np.hstack((X, np.ones((n, 1))))


class LogisticRegression(object):
    """Logistic Regression (aka logit, MaxEnt) classifier.

    In the multiclass case, the training algorithm uses the one-vs-rest (OvR)
    scheme.

    Parameters
    ----------

    solver : str, {'newton', 'gradient-descend'}, optional (default='newton').
        Algorithm to use in the optimization problem.

    max_iter : int, optional (default=100)
        Maximum number of iterations taken for the solvers to converge.

    tol : float, optional (default=1e-8)
        Tolerance for stopping criteria.

    bls_alpha : float, optional (default=0.1)
        Parameter for backtracking line search.

    bls_beta : float, optional (default=0.7)
        Parameter for backtracking line search.

    verbose : int, optional (default=0)
        Any positive number for verbosity.
    """

    def __init__(self, solver='newton', max_iter=100, tol=1e-8,
                 bls_alpha=0.1, bls_beta=0.7, verbose=0):
        self.solver = solver
        self.max_iter = max_iter
        self.tol = tol
        self.bls_alpha = bls_alpha
        self.bls_beta = bls_beta
        self.verbose = verbose

    def fit(self, X, y):
        self.classes_, self.counts = np.unique(y, return_counts=True)
        nclass = self.classes_.size
        if nclass == 2:
            self._fit_binary(X, y)
        elif nclass > 2:
            self._fit_multiclass(X, y)
        else:
            raise ValueError(
                'Incorrect number of target classes: {} detected'.format(nclass))

    def _fit_binary(self, X, y):
        X = _hstack_ones_to_data(X)
        y_binary = _binarize_label(y, self.classes_[1])
        w = self._fit_binary_by_solver(X, y_binary)
        self.coef_ = w[:-1]
        self.intercept_ = w[-1]

    def _fit_multiclass(self, X, y):
        nclass = self.classes_.size
        self.coef_ = np.zeros((nclass, X.shape[1]))
        self.intercept_ = np.zeros(nclass)
        # OvR
        X = _hstack_ones_to_data(X)
        for (i, label) in enumerate(self.classes_):
            y_binary = _binarize_label(y, label)
            w = self._fit_binary_by_solver(X, y_binary)
            self.coef_[i, :] = w[:-1]
            self.intercept_[i] = w[-1]

    def _fit_binary_by_solver(self, X, y):
        if self.solver == 'newton':
            return self._fit_binary_by_newton(X, y)
        if self.solver == 'gradient-descend':
            return self._fit_binary_by_gradient_descend(X, y)
        raise ValueError('Invalid value for solver: ' + self.solver)

    def _fit_binary_by_newton(self, X, y):
        w = np.zeros(X.shape[1])
        for i in range(self.max_iter):
            # Calculate the gradient and hessian of log-likelihood.
            p = sigmoid(X.dot(w))
            gradient = (p - y).dot(X)
            hessian = ((p * (1 - p)) * X.transpose()).dot(X)
            # Calculate the direction: dw = hessian^{-1} * gradient.
            # Attenton: Hessian here is positive definite, therefore, its inverse
            # must exists. However, it can be a zero matrix because p may be
            # rounded to [ones] or [zeros] by error.
            dw = linalg.solve(hessian, gradient, assume_a='pos')
            # Stop criterion: squared_proj / 2 <= self.tol.
            squared_proj = np.inner(dw, gradient)
            if squared_proj / 2 <= self.tol:
                if self.verbose:
                    print('Success: iter = {}'.format(i))
                return w
            # Backtracking line search.
            factor = 1.0
            fw = _log_likelihood(X, y, w)
            while _log_likelihood(X, y, w - factor * dw) \
                    > fw - self.bls_alpha * factor * squared_proj:
                factor *= self.bls_beta
            # Descend.
            w -= factor * dw
        if self.verbose:
            print('Stop iteration on limit {}'.format(self.max_iter))
        return w

    def _fit_binary_by_gradient_descend(self, X, y):
        w = np.zeros(X.shape[1])
        for i in range(self.max_iter):
            # Calculate the gradient of log-likelihood.
            p = sigmoid(X.dot(w))
            gradient = (p - y).dot(X)
            dw = gradient
            # Stop criterion: ||gradient||^2 <= self.tol.
            squared_norm = np.inner(gradient, gradient)
            if squared_norm <= self.tol:
                if self.verbose:
                    print('Success: iter = {}'.format(i))
                return w
            # Backtracking line search.
            factor = 0.1
            fw = _log_likelihood(X, y, w)
            while _log_likelihood(X, y, w - factor * dw) \
                    > fw - self.bls_alpha * factor * squared_norm:
                factor *= self.bls_beta
            # Descend.
            w -= factor * dw
        if self.verbose:
            print('Stop iteration on limit {}'.format(self.max_iter))
        return w

    def predict(self, X):
        nclass = self.classes_.size
        if nclass == 2:
            return self._predict_binary(X)
        elif nclass > 2:
            return self._predict_multiclass(X)
        else:
            raise RuntimeError('This function is called prematurally')

    def _predict_binary(self, X):
        # Calculate the posterior probability: P(y=k|x).
        p = sigmoid(X.dot(self.coef_) + self.intercept_)
        # Rescale the threshold.
        threshold = self.counts[1] / self.counts.sum()
        # Predict.
        mask = (p > threshold)
        p[mask] = self.classes_[1]
        p[np.logical_not(mask)] = self.classes_[0]
        return p

    def _predict_multiclass(self, X):
        # Calculate the posterior probability: p_k = P(y=k|x).
        P = sigmoid(X.dot(self.coef_.transpose()) + self.intercept_)
        # Rescale the probability.
        rescale = self.counts / (self.counts.sum() - self.counts)
        P_rescaled = 1 / (1 + ((1 - P) / P) * rescale)
        # Predict.
        return self.classes_[P_rescaled.argmax(axis=1)]

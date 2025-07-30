import numpy as np
import math


H = np.array([[1, 4, 7], [2, 5, 8], [3, 6, 9], [4, 7, 10]])


def variance_matrix(X):
    column_mean = np.mean(X, axis=0)
    n = len(X)
    var_matrix = (1 / n) * np.diag((X - column_mean).T @ (X - column_mean))
    return var_matrix


def std_matrix(X):
    return np.sqrt(variance_matrix(X))


print(np.var(H, axis=0))
print(variance_matrix(H))


def covariance_matrix(X, Y=None):
    if Y is None:
        Y = X

    X_column_mean = np.mean(X, axis=0)
    Y_column_mean = np.mean(Y, axis=0)

    n = len(X)

    cov_matrix = (1 / (n - 1)) * ((X - X_column_mean).T @ (Y - Y_column_mean))

    return np.array(cov_matrix, dtype=float)


# two np matrices
def correlation_matrix(X, Y=None):

    if Y is None:
        Y = X

    X_column_mean = np.mean(X, axis=0)
    Y_column_mean = np.mean(Y, axis=0)

    n = len(X)

    corr_matrix = (1 / n) * ((X - X_column_mean).T @ (Y - Y_column_mean))

    # corr_matrix = np.divide(corr_matrix, std_matrix(X)@std_matrix(Y))
    print(np.shape(std_matrix(X)))

    std_dev_X = np.expand_dims(std_matrix(X), 1)
    std_dev_y = np.expand_dims(std_matrix(Y), 1)

    print(np.shape(std_dev_X))

    corr_matrix = corr_matrix / std_dev_X.dot(std_dev_y.T)

    return np.array(corr_matrix, dtype=float)


print(np.corrcoef(H, rowvar=False))
print(correlation_matrix(H))


class PCA:

    def __init__(self, n_components):
        self.n_components = n_components

    # covariance =

    def fit_transform(self, X):

        if self.n_components > len(X[0]):
            raise "no"

        cov_matr = covariance_matrix(X)

        eigenvalues, eigenvectors = np.linalg.eig(cov_matr)

        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx][: self.n_components]
        eigenvectors = np.atleast_1d(eigenvectors[:, idx])[:, : self.n_components]

        return X @ eigenvectors

    def transform(self, X):
        pass

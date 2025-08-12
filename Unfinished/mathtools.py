import numpy as np
from sklearn.preprocessing import StandardScaler
import math


def vectorstd(vec):
    n = len(vec)
    mean_vector = np.full(shape=n, fill_value=np.mean(vec))
    return math.sqrt((1 / n) * (np.sum((vec - mean_vector) ** 2)))


column_vec_test = np.array([10, 20, 30])

print(vectorstd(column_vec_test))


def standardize(X):

    for i in range(len(X[0])):
        cur_vec = X[:, i]
        mean_vec = np.full(shape=len(X), fill_value=np.mean(cur_vec))
        X[:, i] = (cur_vec - mean_vec) / vectorstd(cur_vec)

    return X


examplematr = np.array(
    [
        [10, 20, 30],
        [40, 50, 60],
        [70, 80, 90],
    ],
    dtype=float,
)

print(standardize(examplematr))
scaler = StandardScaler()
print(scaler.fit_transform(examplematr))

H = np.array([[1, 4, 7], [2, 5, 8], [3, 6, 9], [4, 7, 10]])  # Example Array


def variance_matrix(X):
    column_mean = np.mean(X, axis=0)
    n = len(X)
    var_matrix = (1 / n) * np.diag((X - column_mean).T @ (X - column_mean))
    return var_matrix


def std_matrix(X):
    return np.sqrt(variance_matrix(X))


def covariance_matrix(X, Y=None):
    if Y is None:
        Y = X
    X_column_mean = np.mean(X, axis=0)
    Y_column_mean = np.mean(Y, axis=0)
    n = len(X)
    cov_matrix = (1 / (n - 1)) * ((X - X_column_mean).T @ (Y - Y_column_mean))
    return np.array(cov_matrix, dtype=float)


def correlation_matrix(X, Y=None):
    if Y is None:
        Y = X

    X_column_mean = np.mean(X, axis=0)
    Y_column_mean = np.mean(Y, axis=0)
    n = len(X)
    corr_matrix = (1 / n) * ((X - X_column_mean).T @ (Y - Y_column_mean))
    std_dev_X = np.expand_dims(std_matrix(X), 1)
    std_dev_y = np.expand_dims(std_matrix(Y), 1)
    corr_matrix = corr_matrix / std_dev_X.dot(std_dev_y.T)

    return np.array(corr_matrix, dtype=float)


# # Tests (Correlation Matr)
# print(np.corrcoef(H, rowvar=False))
# print(correlation_matrix(H))

# # Tests (Var Matr)
# print(np.var(H, axis=0))
# print(variance_matrix(H))



import numpy as np
from Unfinished.mathtools import covariance_matrix
from sklearn.decomposition import PCA
from Unfinished.mathtools import standardize
from sklearn.preprocessing import StandardScaler


class CustomPCA:

    def __init__(self, n_components):
        self.n_components = n_components
        self.eigenvectors = None

    def fit_transform(self, X):

        if self.n_components > len(X[0]):
            raise ValueError("n_components cannot be greater than the number of features in X")
        cov_matr = covariance_matrix(X)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matr)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx][: self.n_components]
        eigenvectors = np.atleast_1d(eigenvectors[:, idx])[:, : self.n_components]
        self.eigenvectors = eigenvectors
        return X @ eigenvectors

    def transform(self, X):
        if self.eigenvectors is None:
            raise ValueError("Haven't fit_transform on a matrix yet")
        elif X.shape[1] != self.eigenvectors.shape[0]:
            raise ValueError("Matrix columns (features) must be equivalent to n_components")
        else:
            return X @ self.eigenvectors


# Tests

TestMatrix = np.array(
    [
        [1.00000001, 2.00000002, 3.00000003],
        [2.00000002, 4.00000004, 6.00000006],
        [3.00000003, 6.00000006, 9.00000009],
        [4.00000004, 8.00000008, 12.00000012],
        [5.00000005, 10.0000001, 15.00000015],
    ]
)

TestMatrix2 = np.array(
    [
        [1.00000001, 2.00000002, 3.00000003],
        [2.00000002, 4.00000004, 6.00000006],
        [3.00000003, 6.00000006, 9.00000009],
        [4.00000004, 8.00000008, 12.00000012],
        [5.00000005, 10.0000001, 15.00000015],
    ]
)

pca1 = PCA(n_components=2)
pca2 = CustomPCA(n_components=2)

scales = StandardScaler()

X_pca1 = pca1.fit_transform(TestMatrix)

X_pca2 = pca2.fit_transform(TestMatrix2)

print(np.round(X_pca1, decimals=2))
print(np.round(X_pca2, decimals=2))

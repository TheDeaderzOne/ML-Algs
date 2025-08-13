import numpy as np
import pandas as pd


class Perceptron:

    def __init__(self, n_iterations=100):
        self.weight_vector = np.array([])
        self.n_iterations = n_iterations
        self.converged = False

    def fit(self, npdata, target):
        # Try to make the target -1, 1
        if len(npdata) != len(target):
            raise ValueError("Must have same amount of target rows (predictions) to data rows (samples)")
        if isinstance(npdata, (pd.DataFrame, pd.Series)):
            npdata = npdata.to_numpy()
        if not isinstance(npdata, (np.ndarray)):
            raise ValueError("Input an np array or a pandas dataframe/series")

        self.weight_vector = np.zeros(len(npdata[0]) + 1)
        ones_col = np.ones((npdata.shape[0], 1))
        npdata = np.hstack((npdata, ones_col))

        for _ in range(self.n_iterations):
            missvar = 0
            for rowind in range(len(npdata)):
                if target[rowind] == 0:
                    sign = -1
                else:
                    sign = 1
                if (sign * (self.weight_vector @ npdata[rowind])) <= 0:
                    self.weight_vector += sign * npdata[rowind]
                    missvar += 1
            if missvar == 0:
                self.converged = True
                break

        if self.converged:
            print("The Perceptron Algorithm has converged")
        else:
            print("The Perceptron Algorithm has hit the iteration limit")

    def predict(self, predictdata):
        if isinstance(predictdata, (pd.DataFrame, pd.Series)):
            predictdata = predictdata.to_numpy()
        if not isinstance(predictdata, (np.ndarray)):
            raise ValueError("Input an np array or a pandas dataframe/series")

        ones_col = np.ones((predictdata.shape[0], 1))
        predictdata = np.hstack((predictdata, ones_col))

        predictions = (self.weight_vector @ predictdata.T > 0).astype(int)

        return predictions

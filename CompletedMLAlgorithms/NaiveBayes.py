import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd
from scipy.stats import norm
from numpy.typing import ArrayLike


# Categorical Naive Bayes
class CategoricalNaiveBayes:
    def __init__(self, smoothing_param: float = 1):
        self.smoothing_param = smoothing_param
        self.data = None
        self.target_classes = 0
        self.target_probabilities = np.array([])
        self.encoder = None
        self.lookup_table = []

    def fit(self, data, target, ordinal_encode: bool = False):

        if len(data) != len(target):
            raise ValueError("Must have same amount of target rows (predictions) to data rows (samples)")

        if ordinal_encode:
            self.encoder = OrdinalEncoder(dtype=int)
            self.encoder.fit(data)
            data = pd.DataFrame(self.encoder.transform(data), columns=data.columns)

        target = target.reset_index(drop=True)

        for x in range(len(data.columns)):
            self.lookup_table.append([])
            unique_values = sorted(data.iloc[:, x].unique())
            for y in range(target.nunique()):
                self.lookup_table[x].append([])
                mask = target == y
                indices = target.index[mask].tolist()
                for z in unique_values:
                    mask2 = data.iloc[indices, x] == z
                    count_of_z = mask2.sum()
                    self.lookup_table[x][y].append(
                        (count_of_z + self.smoothing_param) / (len(indices) + self.smoothing_param * len(unique_values))
                    )

        self.data = data
        self.target_classes = target.nunique()
        self.class_labels = sorted(target.unique())
        self.target_probabilities = np.array([(target == label).mean() for label in self.class_labels])

    def predict(self, testdata):
        if self.data is None:
            raise NotFittedError("This model is not fitted yet. Call 'fit' before 'predict'.")
        if self.encoder is not None:
            testdata_1 = self.encoder.transform(testdata)
            testdata = pd.DataFrame(data=testdata_1, columns=testdata.columns)

        answer_list = np.array([0] * len(testdata))

        def row_predictor(row):
            max_value = 0
            answer = 0

            for y in range(self.target_classes):
                currentval = self.target_probabilities[y]
                for i in range(len(row)):
                    currentval *= self.lookup_table[i][y][row.iloc[i]]
                if currentval > max_value:
                    answer = y
                    max_value = currentval

            return answer

        for i in range(len(testdata)):
            answer_list[i] = row_predictor(testdata.iloc[i])

        return answer_list


class ContinuousNaiveBayes:
    def __init__(self, smoothing_param: float = 1e-9):
        self.smoothing_param = smoothing_param
        self.target_probabilities = np.array([])
        self.target_classes = 0
        self.lookup_table = []

    def fit(self, data, target):

        if len(data) != len(target):
            raise ValueError("Must have same amount of target rows (predictions) to data rows (samples)")

        data = data.reset_index(drop=True)
        target = target.reset_index(drop=True)

        for x in range(len(data.columns)):
            self.lookup_table.append([])
            for y in range(target.nunique()):
                column = data.iloc[:, x]
                mask = target == y
                n = mask.sum()
                estim_mean = column[mask].mean()
                estim_var = self.smoothing_param + (((column[mask] - estim_mean) ** 2).sum() / n)
                self.lookup_table[x].append([estim_mean, estim_var])

        self.target_classes = target.nunique()
        self.class_labels = sorted(target.unique())
        self.target_probabilities = np.array([(target == label).mean() for label in self.class_labels])

    def predict(self, testdata):

        answer_list = np.array([0] * len(testdata))

        def row_predictor(row):
            max_value = -float("inf")
            answer = 0
            for y in range(self.target_classes):
                currentval = np.log(self.target_probabilities[y])
                for i in range(len(row)):
                    mean = self.lookup_table[i][y][0]
                    var = self.lookup_table[i][y][1]
                    hd = norm.pdf(row.iloc[i], mean, np.sqrt(var))
                    currentval += np.log(hd)
                if currentval > max_value:
                    answer = y
                    max_value = currentval

            return answer

        for i in range(len(testdata)):
            answer_list[i] = row_predictor(testdata.iloc[i])

        return answer_list

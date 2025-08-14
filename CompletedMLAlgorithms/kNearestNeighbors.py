import numpy as np
import pandas as pd


class kNN:
    def __init__(self, k=5, power=2):
        self.k = max(1, int(k))
        self.traindata = None
        self.target = None
        self.power = power

    def _validate(self, data):
        if isinstance(data, (pd.DataFrame, pd.Series)):
            data = data.to_numpy()
        if not isinstance(data, (np.ndarray)):
            raise ValueError("Input an np array or a pandas dataframe/series")
        return data

    def fit(self, data, target):
        if len(data) != len(target):
            raise ValueError("Must have same amount of target rows (predictions) to data rows (samples)")
        if isinstance(target, (pd.Series)):
            target = target.to_numpy()
        elif not isinstance(target, (np.ndarray)):
            raise ValueError("Input an np array or a pandas dataframe/series")

        data = np.hstack((self._validate(data), target.reshape(-1, 1)))
        self.traindata = data

    def predict(self, testdata):
        if self.traindata is None:
            raise ValueError("Please fit before you predict")
        testdata = self._validate(testdata)
        answer_list = np.array([0] * len(testdata))

        def distance_func(arr1, arr2, power):
            return (((arr1 - arr2) ** power).sum()) ** (1 / power)

        # minkowski distance
        index = 0
        for predictionrow in testdata:
            keys = np.array([distance_func(row[:-1], predictionrow, power=self.power) for row in self.traindata])
            sorted_arr = self.traindata[np.argsort(keys)][: self.k]
            classifier_column = sorted_arr[:, -1]
            answer_list[index] = pd.Series(classifier_column).mode().iloc[0]
            index += 1

        # compute distance, then sort, then classify the top k

        return answer_list


# kNN does better than perceptron on the same dataset

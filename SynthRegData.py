import random as rand
import torch


class SyntheticRegressionData:
    def __init__(self, weights, bias, noise=0.01, num_train=1000, num_val=1000, batch_size=32):

        # noise helps with "variance"

        self.batch_size = batch_size
        self.weights = weights
        self.bias = bias
        self.noise_level = noise
        self.num_train = num_train
        self.num_val = num_val

        # basically creates the input and output for the training and validations

        totaldatapoints = num_train + num_val

        # Creates tensors of random value, with mean 0 and variance 1 (z-score)
        # Dimension are row * column, so a 2000*2 matrix in a 2d line case

        self.X = torch.randn(totaldatapoints, len(weights))

        # We use (totaldatapoints,1) to denote a vector

        noise = torch.randn(totaldatapoints, 1) * noise

        # the -1 is to denote the "previous" dimensions

        # This creates the "answer" dataset for the randomly generated noise points

        self.y = self.X @ weights.reshape((-1, 1)) + bias + noise

    def get_dataloader(self, train: bool):
        if train:
            indices = [int(x) for x in range(self.num_train)]
            rand.shuffle(indices)
        else:
            indices = [int(x) for x in range(self.num_train, self.num_train + self.num_val)]

        for i in range(0, len(indices), self.batch_size):

            # redundant tensor conversion
            batch_indices = torch.tensor(indices[i : i + self.batch_size])

            yield self.X[batch_indices], self.y[batch_indices]

    def train_dataloader(self):
        return self.get_dataloader(train=True)

    def val_dataloader(self):
        return self.get_dataloader(train=False)

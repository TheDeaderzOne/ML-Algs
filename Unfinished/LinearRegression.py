import torch

# import torch.nn as nn
from Unfinished.TrainerClass import Trainer
from Unfinished.Module import ModelModule
from Unfinished.SynthRegData import SyntheticRegressionData
import matplotlib.pyplot as plt


class StochasticGradientDescent:
    # params is a collection of tensors, like the weight, and the bias
    # How to require type in python
    def __init__(self, params, step_size):
        self.params = params
        self.step_size = step_size

    def step(self):
        for miniparam in self.params:
            miniparam -= self.step_size * miniparam.grad

    # zeroes all the gradients
    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()


class LinReg(ModelModule):

    def __init__(self, inputs, step_size, sigma=0.01):
        super().__init__()
        self.inputs = inputs
        self.step_size = step_size
        self.sigma = sigma
        # weighting system, where there is some "randomness" introduced with torch.normal
        self.weights = torch.normal(0, sigma, (inputs, 1), requires_grad=True)
        self.bias = torch.zeros(1, requires_grad=True)

    def forward(self, X):
        return X @ self.weights + self.bias

    # These are all "vectors"/tensors of (batch_size,1)
    def loss(self, y_hat, y):
        l = (y_hat - y) ** 2 / 2
        return l.mean()

    def configure_optimizers(self):
        return StochasticGradientDescent([self.weights, self.bias], self.step_size)


model = LinReg(2, step_size=0.03)
data = SyntheticRegressionData(weights=torch.tensor([2, -3.4]), bias=4.2)
trainer = Trainer(max_epochs=3)
trainer.fit(model, data)


plt.show()
plt.close()

import numpy as np


class Layer:
    def forward(self, inputs):
        raise NotImplementedError

    def backward(self, dLoss_dOutput):
        raise NotImplementedError


class Dense(Layer):
    def __init__(self, nbOfInputs: int, nbOfOutputs: int) -> None:
        self.nbOfInputs = nbOfInputs
        self.nbOfOutputs = nbOfOutputs

        # initialize
        self.biases = np.zeros(nbOfOutputs)
        self.weights = np.random.randn(nbOfInputs, nbOfOutputs) * np.sqrt(2 / nbOfInputs)

    def forward(self, X):
        self.output = np.dot(X, self.weights) + self.biases
        self.inputs = X

    def backward(self, dLoss_dNeurons):
        # compute gradients
        self.dLoss_dWeights = np.dot(self.inputs.T, dLoss_dNeurons)
        self.dLoss_dBiases = np.average(dLoss_dNeurons, axis=0)  # condense to 1xm dimensions
        self.dLoss_dInputs = np.dot(dLoss_dNeurons, self.weights.T)


class Dropout(Layer):
    def __init__(self, rate: float):
        self.rate = rate

    def forward(self, inputs):
        self.binomial = np.random.binomial(1, 1-self.rate, inputs.shape) / (1 - self.rate)
        self.output = inputs * self.binomial

    def backward(self, dLoss_dOutput):
        self.dLoss_dInputs = dLoss_dOutput * self.binomial


class Flatten(Layer):
    def forward(self, inputs):
        if len(inputs.shape) <= 1: return inputs
        return np.reshape(inputs, (len(inputs), np.prod(inputs.shape[1:])))
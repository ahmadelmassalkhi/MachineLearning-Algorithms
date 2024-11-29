import numpy as np


class Layer:
    def __init__(self, nbOfInputs:int, nbOfOutputs:int) -> None:
        self.lr = 0.01 # constant learning rate
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
        self.dLoss_dBiases = np.average(dLoss_dNeurons, axis=0) # condense to 1xm dimensions
        self.dLoss_dInputs = np.dot(dLoss_dNeurons, self.weights.T)
        
        # descent gradients
        self.weights -= self.lr * self.dLoss_dWeights
        self.biases -= self.lr * self.dLoss_dBiases

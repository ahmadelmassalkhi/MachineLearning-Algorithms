import numpy as np
from Activation import ReLU, Softmax
from Loss import CategoricalCrossEntropy

class Softmax_CategoricalCrossEntropy:
    def __init__(self) -> None:
        self.softmax = Softmax()
        self.loss = CategoricalCrossEntropy()

    def forward(self, inputs, y_real):
        self.y_real = y_real
        self.softmax.forward(inputs)
        self.loss.forward(self.softmax.output, y_real)

    def backward(self):
        # make sure y_real is one hot encoded
        if self.y_real.shape != self.softmax.output.shape:
            self.y_real = np.eye(self.softmax.output.shape[1])[self.y_real]
        self.dLoss_dInputs = (self.softmax.output - self.y_real) / len(self.y_real)
        

class Layer:
    def __init__(self, nbOfInputs:int, nbOfOutputs:int) -> None:
        self.nbOfInputs = nbOfInputs
        self.nbOfOutputs = nbOfOutputs
        self.lr = 0.01 # constant learning rate
        
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

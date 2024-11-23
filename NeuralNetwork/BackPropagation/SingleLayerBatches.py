import numpy as np
np.random.seed(0)
import time




def relu(x):
    return np.maximum(x,0)

def relu_derivative(x):
    return np.where(x>0, 1,0)


class Layer_Dense:
    def __init__(self, nbOfInputs:int, nbOfOutputs:int) -> None:
        self.nbOfInputs = nbOfInputs
        self.nbOfOutputs = nbOfOutputs

        # initialize
        self.biases = np.zeros(nbOfOutputs)
        self.weights = 0.01 * np.random.randn(nbOfInputs, nbOfOutputs)

    def forward(self, X:np.array):
        if len(X.shape) > 1 and X.shape[1] != self.nbOfInputs:
            raise ValueError(f'Input X must have dimensions: (any, {self.nbOfInputs})')
        self.output = np.dot(X, self.weights) + self.biases
        return self.output
    
    def backward(self, X:np.array, expected_output:np.array, lr=0.0001, n_iterations=2000):
        nbOfBatches, nbOfInputs = X.shape
        if nbOfBatches != expected_output.shape[0]:
            raise ValueError(f'Number of outputs ({expected_output.shape[0]}) must = Number of inputs ({nbOfBatches}) !')
        
        for i in range(n_iterations):
            ReLU = relu(self.forward(X))
            output = np.sum(ReLU, axis=1, keepdims=True)
            loss = (output - expected_output) ** 2
            print(f'Iteration {i+1}')
            # print(f'Loss = {loss}')
            print(f'Output = {output}')
            print()

            # compute gradients
            dLoss_dLinear = 2 * (output-expected_output) * relu_derivative(ReLU)
            dLoss_dWeights = np.dot(X.T, dLoss_dLinear)
            dLoss_dBiases = np.sum(dLoss_dLinear, axis=0)

            # descent gradients
            self.weights -= lr * dLoss_dWeights
            self.biases -= lr * dLoss_dBiases
        # print stats
        print()
        print(f'Final Output = {output}')
        print(f'Final Weights = {self.weights}')
        print(f'Final Biases = {self.biases}')

inputs = np.array([[1,2,3,4],
                   [5,6,7,-8],
                   [9,-10,11,12]])

layer = Layer_Dense(inputs.shape[1], 4)
layer.backward(inputs, expected_output=np.array([[30.52],[6],[9]]))
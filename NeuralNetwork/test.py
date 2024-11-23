import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()


n_classes = 3 # number of different outputs (example: cat, dog, human...)
batches, expected_outputs = spiral_data(2, n_classes)


class NeuralNetwork:
    def __init__(self) -> None:
        pass


class Layer:
    def __init__(self, nbOfInputs:int, nbOfOutputs:int) -> None:
        self.nbOfInputs = nbOfInputs
        self.nbOfOutputs = nbOfOutputs

        # initialize
        self.biases = np.zeros(nbOfOutputs)
        self.weights = 0.01 * np.random.randn(nbOfInputs, nbOfOutputs)

    def forward(self, X:np.array):
        if len(X.shape)>1 and X.shape[1] != self.nbOfInputs:
            raise ValueError(f'Input X must have dimensions: (any, {self.nbOfInputs})')
        self.output = np.dot(X, self.weights) + self.biases
        return self.output
    

layer1 = Layer(batches.shape[1], 3)
layer1.forward(batches)

layer2 = Layer(layer1.output.shape[1], 4)
layer2.forward(layer1.output)

layerN = Layer(layer2.output.shape[1], 1)
layerN.forward(layer2.output)

print(f'Last layer\'s output = {layerN.output}')
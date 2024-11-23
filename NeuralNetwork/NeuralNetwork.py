import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()


n_classes = 3 # number of different outputs (example: cat, dog, human...)
batches, real_outputs = spiral_data(4, n_classes)


class NeuralNetwork:
    def __init__(self) -> None:
        pass


class Layer_Dense:
    def __init__(self, nbOfInputs:int, nbOfOutputs:int) -> None:
        self.nbOfInputs = nbOfInputs
        self.nbOfOutputs = nbOfOutputs

        # initialize
        self.biases = np.zeros(nbOfOutputs)
        self.weights = 0.01 * np.random.randn(nbOfInputs, nbOfOutputs)

    def forward(self, X:np.array):
        if X.shape[1] != self.nbOfInputs:
            raise ValueError(f'Input X must have dimensions: (any, {self.nbOfInputs})')
        self.output = np.dot(X, self.weights) + self.biases
        return self.output
    

class Activation_Softmax:
    def forward(self, input):
        exp = np.exp(input - np.max(input, axis=1, keepdims=True))
        self.output = exp / np.sum(exp, axis=1, keepdims=True)
        return self.output


class Activation_ReLU:
    def forward(self, input):
        self.output = np.maximum(0, input)
        return self.output
    

class Loss:
    def calculate(self, y_pred:np.array, y_real:np.array):
        return np.mean(self.forward(y_pred, y_real))
    
class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_pred:np.array, y_real:np.array):
        # incase of one hot encoding
        if y_real.shape == y_pred.shape:
            confidence = np.sum(y_real * y_pred)
        
        # incase of one list
        elif len(y_real.shape)==1 and y_real.shape[0]==y_pred.shape[0]:
            confidence = y_pred[range(len(y_pred)), y_real]

        else: raise ValueError('Incompatible shapes of predicted:real as {y_real.shape}:{output_prob.shape}')
        
        # compute negative log likelihood
        return -np.log(np.clip(confidence, 1e-7, 1-1e-7))


layer = Layer_Dense(2, 3)
layer.forward(batches)

layer2 = Layer_Dense(3, 3)
layer2.forward(Activation_ReLU().forward(layer.output))

softmax_output = Activation_Softmax().forward(layer2.output)


loss = Loss_CategoricalCrossEntropy().calculate(softmax_output, real_outputs)
print(f'Loss = {loss}')

accuracy = np.mean(real_outputs == np.argmax(softmax_output, axis=1))
print(f'Accuracy = {accuracy}')
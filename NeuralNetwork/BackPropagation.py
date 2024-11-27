import numpy as np
from Activation import ReLU, Softmax
from Loss import CategoricalCrossEntropy

##################################################################

class Softmax_CategoricalCrossEntropy:
    def __init__(self) -> None:
        self.softmax = Softmax()
        self.loss = CategoricalCrossEntropy()

    def forward(self, inputs, y_real):
        self.y_real = y_real
        self.softmax.forward(inputs)
        self.loss.forward(self.softmax.output, y_real)

    def backward(self):
        # y_real is expected to be one hot encoded
        if self.y_real.shape != self.softmax.output.shape:
            self.y_real = np.eye(self.softmax.output.shape[1])[self.y_real]
        self.dLoss_dInputs = (self.softmax.output - self.y_real) / len(self.y_real)
        
##################################################################

class Layer:
    def __init__(self, nbOfInputs:int, nbOfOutputs:int) -> None:
        self.nbOfInputs = nbOfInputs
        self.nbOfOutputs = nbOfOutputs
        self.lr = 0.01
        
        # initialize
        self.biases = np.zeros(nbOfOutputs)
        self.weights = np.random.randn(nbOfInputs, nbOfOutputs) * np.sqrt(2 / nbOfInputs)

    def forward(self, X):
        self.output = np.dot(X, self.weights) + self.biases
        self.inputs = X
    
    def backward(self, dLoss_dNeurons):
        self.dLoss_dWeights = np.dot(self.inputs.T, dLoss_dNeurons)
        self.dLoss_dBiases = np.average(dLoss_dNeurons, axis=0) # condense to 1xm dimensions
        self.dLoss_dInputs = np.dot(dLoss_dNeurons, self.weights.T)
        self.weights -= self.lr * self.dLoss_dWeights
        self.biases -= self.lr * self.dLoss_dBiases

np.random.seed(0)
y_real = np.array([2,0,1])
batches = np.array([[1,2,-3,4],
                    [5,6,7,-8],
                    [9,-10,11,12]])

##################################################################

layer1 = Layer(len(batches[0]), 3)
layer2 = Layer(layer1.nbOfOutputs, 4)
layerN = Layer(layer2.nbOfOutputs, 4)
activation1 = ReLU()
activation2 = ReLU()
Softmax_Loss = Softmax_CategoricalCrossEntropy()

##################################################################

n_iterations = 1000
for i in range(n_iterations):
    ####################
    ''' FORWARD PASS '''
    
    layer1.forward(batches)
    activation1.forward(layer1.output)

    layer2.forward(activation1.output)
    activation2.forward(layer2.output)

    layerN.forward(activation2.output)
    Softmax_Loss.forward(layerN.output, y_real)

    print(f'Iteration {i+1}, Confidences Of Correct Classes = {Softmax_Loss.softmax.output[range(len(batches)), y_real]}')
    #####################
    ''' BACKWARD PASS '''

    Softmax_Loss.backward()
    layerN.backward(Softmax_Loss.dLoss_dInputs)

    activation2.backward(layerN.dLoss_dInputs)
    layer2.backward(activation2.dLoss_dInputs)

    activation1.backward(layer2.dLoss_dInputs)
    layer1.backward(activation1.dLoss_dInputs)
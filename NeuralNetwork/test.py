import numpy as np
from Activation import ReLU
from BackPropagation import Layer, Softmax_CategoricalCrossEntropy

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
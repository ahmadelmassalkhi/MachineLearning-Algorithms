import nnfs.datasets
from lib.Activation import ReLU, Softmax
from lib.Layer import Layer
from lib.Loss_Activation import Softmax_CategoricalCrossEntropy
import nnfs
import numpy as np
from lib.Optimizer import Optimizer_GD


# init dataset
nbOfInputs = 2 # always true for spiral dataset 
n_samples, n_classes = 4, 3
(X_train, y_train) = nnfs.datasets.spiral_data(n_samples, n_classes) 


# init model
layer1 = Layer(nbOfInputs, 64)
layer2 = Layer(layer1.nbOfOutputs, n_classes)
activation1 = ReLU()
activation_loss = Softmax_CategoricalCrossEntropy()
optimizer = Optimizer_GD()


# train model
for i in range(10001):
    ''' FORWARD PASS '''
    layer1.forward(X_train)
    activation1.forward(layer1.output)

    layer2.forward(activation1.output)
    activation_loss.forward(layer2.output, y_train)

    ''' EVALUATE MODEL PREDICTION '''
    loss = np.average(activation_loss.loss.output)
    accuracy = np.mean(np.argmax(activation_loss.softmax.output, axis=1) == y_train)
    if not i % 10:
        print(f'epoch {i} '+
              f'Loss = {loss} ' +
              f'Accuracy = {accuracy}')
    
    ''' BACKWARD PASS '''
    activation_loss.backward()
    layer2.backward(activation_loss.dLoss_dInputs)

    activation1.backward(layer2.dLoss_dInputs)
    layer1.backward(activation1.dLoss_dInputs)

    ''' OPTIMIZE '''
    optimizer.update_params(layer1)
    optimizer.update_params(layer2)
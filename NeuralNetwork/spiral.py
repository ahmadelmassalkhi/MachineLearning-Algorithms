import nnfs.datasets
from lib.Activation import ReLU, Softmax
from lib.Layer import Layer
from lib.Loss_Activation import Softmax_CategoricalCrossEntropy
import nnfs
import numpy as np
from lib.Optimizer import SGD


# init dataset
nbOfInputs = 2 # always true for spiral dataset 
n_samples, n_classes = 100, 3
(X_train, y_train) = nnfs.datasets.spiral_data(n_samples, n_classes) 


# init model
layer1 = Layer(nbOfInputs, 64)
layer2 = Layer(layer1.nbOfOutputs, n_classes)
activation1 = ReLU()
activation_loss = Softmax_CategoricalCrossEntropy()
optimizer = SGD(decay=1e-3, momentum=0.9)


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
    if not i % 100:
        print(f'epoch {i}, '+
              f'acc: {accuracy}, ' +
              f'loss: {loss}, ' +
              f'lr: {optimizer.current_lr}')
    
    ''' BACKWARD PASS '''
    activation_loss.backward()
    layer2.backward(activation_loss.dLoss_dInputs)

    activation1.backward(layer2.dLoss_dInputs)
    layer1.backward(activation1.dLoss_dInputs)

    ''' OPTIMIZE '''
    optimizer.update_params(layer1, i)
    optimizer.update_params(layer2, i)



''' STUCK IN LOCAL MINIMA '''
# GRADIENT DESCENT:
# epoch 10000, acc: 0.48, loss: 1.0185533744272335, lr: 1.0

# GRADIENT DESCENT + DECAY:
# epoch 10000, acc: 0.5066666666666667, loss: 0.9851354937656096, lr: 0.09091735612328393

# GRADIENT DESCENT + DECAY + MOMENTUM
# epoch 10000, acc: 0.6666666666666666, loss: 0.709353446293288, lr: 0.09091735612328393
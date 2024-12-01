import numpy as np
import nnfs
import nnfs.datasets
from lib.Layer import Layer
from lib.Activation import *
from lib.Loss_Activation import Softmax_CategoricalCrossEntropy
from lib.Optimizer import *
nnfs.init()


# init dataset
nbOfInputs = 2 # always true for spiral dataset 
n_samples, n_classes = 100, 3
(X_train, y_train) = nnfs.datasets.spiral_data(n_samples, n_classes) 


# init model
layer1 = Layer(nbOfInputs, 64)
layer2 = Layer(layer1.nbOfOutputs, n_classes)
activation1 = ReLU()
activation_loss = Softmax_CategoricalCrossEntropy()
# optimizer = SGD(lr=1) # acc: 0.610
# optimizer = SGD(lr=0.1, decay=1e-6) # acc: 0.483
# optimizer = SGD(lr=0.1, decay=1e-7, momentum=0.99) # acc: 0.983
# optimizer = AdaGrad(lr=0.1, decay=1e-7) # acc: 0.837
optimizer = RMSProp(lr=0.1, decay=1e-2, rho=0.9) # acc: 0.953



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
              f'acc: {accuracy:.3f}, ' +
              f'loss: {loss:.3f}, ' +
              f'lr: {optimizer.current_lr}')
    
    ''' BACKWARD PASS '''
    activation_loss.backward()
    layer2.backward(activation_loss.dLoss_dInputs)

    activation1.backward(layer2.dLoss_dInputs)
    layer1.backward(activation1.dLoss_dInputs)

    ''' OPTIMIZE '''
    optimizer.pre_update_params()
    optimizer.update_params(layer1)
    optimizer.update_params(layer2)
    optimizer.post_update_params()


''' STUCK IN LOCAL MINIMA '''

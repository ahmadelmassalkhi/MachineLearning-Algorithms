import numpy as np
from helper_functions import *
from lib.Layer import Layer
from lib.Activation import ReLU
from lib.Loss_Activation import Softmax_CategoricalCrossEntropy
import pickle
# from tensorflow.keras.datasets import mnist # type: ignore


##################################################################
''' LOAD TRAINING DATASET '''

# (X_train, y_train), (X_test, y_test) = mnist_dataset()
X_train = np.empty((0, 28, 28))  # Start with an empty array
y_train = np.empty((0,), dtype=int)  # Ensure y_train is an integer array
for i in range(10):
    a, b = predict_all_images(i)
    X_train = np.concatenate((X_train, a), axis=0)  # Concatenate along the first axis
    y_train = np.concatenate((y_train, b), axis=0)  # Concatenate labels
X_train = X_train.reshape(X_train.shape[0], 28*28)  # Flatten the images

##################################################################
''' LOAD MODEL '''

activation1 = ReLU()
activation2 = ReLU()
Softmax_Loss = Softmax_CategoricalCrossEntropy()

try:
    filename = 'model.pkl'
    # Attempt to open and load the model (layers)
    with open(filename, 'rb') as file:
        layer1, layer2, layerN = pickle.load(file)
    print(f'Model loaded from {filename}')
except FileNotFoundError:
    # If the model file doesn't exist, initialize the layers
    layer1 = Layer(len(X_train[0]), 256)
    layer2 = Layer(layer1.nbOfOutputs, 128)
    layerN = Layer(layer2.nbOfOutputs, 10)
    print("Model initialized from scratch")

##################################################################

confidence = i = 0
while confidence < 0.99:
    ####################
    ''' FORWARD PASS '''
    
    layer1.forward(X_train)
    activation1.forward(layer1.output)

    layer2.forward(activation1.output)
    activation2.forward(layer2.output)

    layerN.forward(activation2.output)
    Softmax_Loss.forward(layerN.output, y_train)

    confidence = np.average(Softmax_Loss.softmax.output[range(len(X_train)), y_train])
    print(f'Iteration {i+1}, Average Correct Confidence = {confidence}')
    #####################
    ''' BACKWARD PASS '''

    Softmax_Loss.backward()
    layerN.backward(Softmax_Loss.dLoss_dInputs)

    activation2.backward(layerN.dLoss_dInputs)
    layer2.backward(activation2.dLoss_dInputs)

    activation1.backward(layer2.dLoss_dInputs)
    layer1.backward(activation1.dLoss_dInputs)

    i += 1

##################################################################
''' SAVE MODEL '''

# Save layers to a file (to save their weights & biases)
with open('model.pkl', 'wb') as file:
    pickle.dump([layer1, layer2, layerN], file)
    print("Model saved to model.pkl")


import pickle
from lib.Activation import ReLU
from lib.Loss_Activation import Softmax_CategoricalCrossEntropy
from lib.Layer import Layer
from tensorflow.keras.datasets import mnist # type: ignore
import numpy as np

NB_OF_BATCHES = 6000

##################################################################
''' PREPARE TRAINING DATASET '''

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# keep last 10 samples
X_train = X_train[-NB_OF_BATCHES::] # (6000, 28, 28) => (batches, 28, 28)
y_train = y_train[-NB_OF_BATCHES:] # (6000,) => (batches,)

# Flatten the images (batches, 28, 28) => (batches, 28*28) => (batches, 784)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])

# Normalize pixel values to [0, 1] for better performance
X_train = X_train.astype('float32') / 255.0

##################################################################
''' PREPARE MODEL '''

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

n_iterations = 10000
for i in range(n_iterations):
    ####################
    ''' FORWARD PASS '''
    
    layer1.forward(X_train)
    activation1.forward(layer1.output)

    layer2.forward(activation1.output)
    activation2.forward(layer2.output)

    layerN.forward(activation2.output)
    Softmax_Loss.forward(layerN.output, y_train)

    # print(f'Iteration {i+1}, Confidences Of Correct Classes = {Softmax_Loss.softmax.output[range(len(X_train)), y_train]}')
    # print(f'Iteration {i+1}, Predicted Numbers = {np.argmax(Softmax_Loss.softmax.output, axis=1)}')
    # print(f'Confidences Of Correct Classes = {Softmax_Loss.softmax.output[range(len(X_train)), y_train]}')
    # print(f'Predicted Numbers = {np.argmax(Softmax_Loss.softmax.output, axis=1)}')
    # print(f'Iteration {i+1}, Average Loss = {np.average(Softmax_Loss.loss.output)}')
    print(f'Iteration {i+1}, Average Confidence = {np.average(Softmax_Loss.softmax.output[range(len(X_train)), y_train])}')
    #####################
    ''' BACKWARD PASS '''

    Softmax_Loss.backward()
    layerN.backward(Softmax_Loss.dLoss_dInputs)

    activation2.backward(layerN.dLoss_dInputs)
    layer2.backward(activation2.dLoss_dInputs)

    activation1.backward(layer2.dLoss_dInputs)
    layer1.backward(activation1.dLoss_dInputs)


##################################################################
''' SAVE MODEL '''

# Save layers to a file (to save their weights & biases)
with open('model.pkl', 'wb') as file:
    pickle.dump([layer1, layer2, layerN], file)
    print("Model saved to model.pkl")


import numpy as np
from helper_functions import *
from lib.Layer import Layer
from lib.Activation import ReLU, Softmax
import pickle
# from tensorflow.keras.datasets import mnist # type: ignore


####################################################################
''' PREPARE DATASET '''

# (X_train, y_train), (X_test, y_test) = mnist_dataset()
X_train = predict_png_image()
X_train_flattened = X_train.reshape(X_train.shape[0], 28*28)  # Flatten the images

####################################################################
''' LOAD MODEL '''
try:
    filename = 'model.pkl'
    # Attempt to open and load the model (layers)
    with open(filename, 'rb') as file:
        layer1, layer2, layerN = pickle.load(file)
    print(f'Model loaded from {filename}')
except FileNotFoundError:
    # If the model file doesn't exist, initialize the layers
    layer1 = Layer(X_train_flattened.shape[-1], 256)
    layer2 = Layer(layer1.nbOfOutputs, 128)
    layerN = Layer(layer2.nbOfOutputs, 10)
    print("Model initialized from scratch")

activation1 = ReLU()
activation2 = ReLU()
activationN = Softmax()

####################################################################
''' FORWARD PASS '''
layer1.forward(X_train_flattened)
activation1.forward(layer1.output)

layer2.forward(activation1.output)
activation2.forward(layer2.output)

layerN.forward(activation2.output)
activationN.forward(layerN.output)

####################################################################
''' PRINT PREDICTION '''

prediction = np.argmax(activationN.output[0])
print(f'Confidence = {activationN.output[0][prediction]}')
plot_image(X_train[0], prediction)

# print(f'Average Correct Confidence = {np.average(activationN.output[range(len(y_train)), y_train])}')


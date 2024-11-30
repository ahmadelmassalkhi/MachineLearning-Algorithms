import pickle
import numpy as np
from lib.Layer import Layer
import matplotlib.pyplot as plt
from lib.Activation import ReLU, Softmax
from lib.Loss_Activation import Softmax_CategoricalCrossEntropy
from tensorflow.keras.datasets import mnist  # type: ignore
from lib.digit_recognition import NeuralNetwork


def mnist_dataset(batchSize:int=100):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # choose batch size
    X_train = X_train[-batchSize::] # (60000, 28, 28) => (batches, 28, 28)
    y_train = y_train[-batchSize:] # (60000,) => (batches,)

    # Normalize pixel values to [0, 1] for WAY better performance
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    return (X_train, y_train), (X_test, y_test)

(X_train, y_train), (X_test, y_test) = mnist_dataset()
X_train = X_train.reshape(len(X_train), 28*28)

nn = NeuralNetwork(model_path='model.pkl')
nn.learn(X_train, y_train)
nn.save()

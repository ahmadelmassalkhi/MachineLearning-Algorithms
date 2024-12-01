from lib.Layer import Layer
import numpy as np


class SGD:
    def __init__(self, lr=1, decay=0, momentum=0) -> None:
        self.lr = self.current_lr = lr
        self.decay = decay
        self.momentum = momentum

    def update_params(self, layer:Layer, iteration:int):
        self.current_lr = (self.lr) / (1 + self.decay * iteration)

        if not hasattr(layer, 'weights_momentums'):
            layer.weights_momentums = np.zeros_like(layer.weights)
            layer.biases_momentums = np.zeros_like(layer.biases)

        # Update momentums
        layer.weights_momentums = self.momentum * layer.weights_momentums - self.current_lr * layer.dLoss_dWeights
        layer.biases_momentums = self.momentum * layer.biases_momentums - self.current_lr * layer.dLoss_dBiases

        # Use momentums for updates
        layer.weights += layer.weights_momentums
        layer.biases += layer.biases_momentums


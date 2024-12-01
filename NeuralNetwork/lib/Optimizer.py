from lib.Layer import Layer
import numpy as np


class Optimizer:
    def update_params(self):
        pass

    def update_network(self, layers:list[Layer]):
        if hasattr(self, 'pre_update_params'): self.pre_update_params()
        for layer in layers:
            self.update_params(layer)
        if hasattr(self, 'post_update_params'): self.post_update_params()


class SGD(Optimizer):
    def __init__(self, lr=1, decay=0, momentum=0) -> None:
        self.lr = self.current_lr = lr
        self.iteration = 0
        self.decay = decay
        self.momentum = momentum

    def pre_update_params(self):
        self.current_lr = (self.lr) / (1 + self.decay * self.iteration)

    def update_params(self, layer:Layer):
        if not hasattr(layer, 'weights_momentums'):
            layer.weights_momentums = np.zeros_like(layer.weights)
            layer.biases_momentums = np.zeros_like(layer.biases)

        # Update momentums
        layer.weights_momentums = self.momentum * layer.weights_momentums - self.current_lr * layer.dLoss_dWeights
        layer.biases_momentums = self.momentum * layer.biases_momentums - self.current_lr * layer.dLoss_dBiases

        # Use momentums for updates
        layer.weights += layer.weights_momentums
        layer.biases += layer.biases_momentums

    def post_update_params(self):
        self.iteration += 1




class AdaGrad(Optimizer):
    def __init__(self, lr=1, decay=0, epsilon=1e-7) -> None:
        self.lr = self.current_lr = lr
        self.iteration = 0
        self.decay = decay
        self.epsilon = epsilon

    def pre_update_params(self):
        self.current_lr = (self.lr) / (1 + self.decay * self.iteration)

    def update_params(self, layer:Layer):
        if not hasattr(layer, 'weights_accumulative_gradients'):
            layer.weights_accumulative_gradients = np.zeros_like(layer.weights)
            layer.biases_accumulative_gradients = np.zeros_like(layer.biases)
        
        layer.weights_accumulative_gradients += layer.dLoss_dWeights ** 2
        layer.biases_accumulative_gradients += layer.dLoss_dBiases ** 2

        layer.weights -= self.current_lr * layer.dLoss_dWeights / np.sqrt(self.epsilon + layer.weights_accumulative_gradients)
        layer.biases -= self.current_lr * layer.dLoss_dBiases / np.sqrt(self.epsilon + layer.biases_accumulative_gradients)

    def post_update_params(self):
        self.iteration += 1


class RMSProp(Optimizer):
    def __init__(self, lr=1, decay=0, epsilon=1e-7, rho=0.9) -> None:
        self.lr = self.current_lr = lr
        self.iteration = 0
        self.decay = decay
        self.epsilon = epsilon
        self.rho = rho

    def pre_update_params(self):
        self.current_lr = (self.lr) / (1 + self.decay * self.iteration)

    def update_params(self, layer:Layer):
        if not hasattr(layer, 'weights_accumulative_gradients'):
            layer.weights_accumulative_gradients = np.zeros_like(layer.weights)
            layer.biases_accumulative_gradients = np.zeros_like(layer.biases)
        
        layer.weights_accumulative_gradients = self.rho * layer.weights_accumulative_gradients + (1-self.rho)*layer.dLoss_dWeights ** 2
        layer.biases_accumulative_gradients = self.rho * layer.biases_accumulative_gradients + (1-self.rho)*layer.dLoss_dBiases ** 2

        layer.weights -= self.current_lr * layer.dLoss_dWeights / np.sqrt(self.epsilon + layer.weights_accumulative_gradients)
        layer.biases -= self.current_lr * layer.dLoss_dBiases / np.sqrt(self.epsilon + layer.biases_accumulative_gradients)

    def post_update_params(self):
        self.iteration += 1
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
        # update momentums
        if not hasattr(layer, 'weights_momentums'):
            layer.weights_momentums = np.zeros_like(layer.weights)
            layer.biases_momentums = np.zeros_like(layer.biases)
        layer.weights_momentums = self.momentum * layer.weights_momentums - self.current_lr * layer.dLoss_dWeights
        layer.biases_momentums = self.momentum * layer.biases_momentums - self.current_lr * layer.dLoss_dBiases

        # optimize
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
        # update accumulative gradients
        if not hasattr(layer, 'weights_accumulative_gradients'):
            layer.weights_accumulative_gradients = np.zeros_like(layer.weights)
            layer.biases_accumulative_gradients = np.zeros_like(layer.biases)
        layer.weights_accumulative_gradients += layer.dLoss_dWeights ** 2
        layer.biases_accumulative_gradients += layer.dLoss_dBiases ** 2

        # optimize
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
        # update accumulative gradients
        if not hasattr(layer, 'weights_accumulative_gradients'):
            layer.weights_accumulative_gradients = np.zeros_like(layer.weights)
            layer.biases_accumulative_gradients = np.zeros_like(layer.biases)
        layer.weights_accumulative_gradients = self.rho * layer.weights_accumulative_gradients + (1-self.rho)*layer.dLoss_dWeights ** 2
        layer.biases_accumulative_gradients = self.rho * layer.biases_accumulative_gradients + (1-self.rho)*layer.dLoss_dBiases ** 2

        # optimize
        layer.weights -= self.current_lr * layer.dLoss_dWeights / np.sqrt(self.epsilon + layer.weights_accumulative_gradients)
        layer.biases -= self.current_lr * layer.dLoss_dBiases / np.sqrt(self.epsilon + layer.biases_accumulative_gradients)

    def post_update_params(self):
        self.iteration += 1


class Adam(Optimizer):
    def __init__(self, lr=0.01, decay=0, epsilon=1e-7, beta1=0.9, beta2=0.999):
        self.lr = self.current_lr = lr  # Base learning rate
        self.iteration = 0  # Tracks the number of updates
        self.decay = decay  # Learning rate decay factor
        self.epsilon = epsilon  # Small value to prevent division by zero
        self.beta1 = beta1  # Decay rate for momentums
        self.beta2 = beta2  # Decay rate for cache

    def pre_update_params(self):
        # Adjust learning rate based on decay
        self.current_lr = self.lr / (1 + self.decay * self.iteration)

    def update_params(self, layer: Layer):
        # Initialize momentum and cache if not already set
        if not hasattr(layer, 'm_w'):
            layer.m_w = np.zeros_like(layer.weights)
            layer.m_b = np.zeros_like(layer.biases)
            layer.v_w = np.zeros_like(layer.weights)
            layer.v_b = np.zeros_like(layer.biases)

        # Update momentum (exponential moving average of gradients)
        layer.m_w = self.beta1 * layer.m_w + (1 - self.beta1) * layer.dLoss_dWeights
        layer.m_b = self.beta1 * layer.m_b + (1 - self.beta1) * layer.dLoss_dBiases

        # Update cache (exponential moving average of squared gradients)
        layer.v_w = self.beta2 * layer.v_w + (1 - self.beta2) * (layer.dLoss_dWeights ** 2)
        layer.v_b = self.beta2 * layer.v_b + (1 - self.beta2) * (layer.dLoss_dBiases ** 2)

        # Bias correction for momentum and cache
        m_w_corr = layer.m_w / (1 - self.beta1 ** (self.iteration + 1))
        m_b_corr = layer.m_b / (1 - self.beta1 ** (self.iteration + 1))
        v_w_corr = layer.v_w / (1 - self.beta2 ** (self.iteration + 1))
        v_b_corr = layer.v_b / (1 - self.beta2 ** (self.iteration + 1))

        # normalize gradients
        grad_w = m_w_corr / (np.sqrt(v_w_corr) + self.epsilon)
        grad_b = m_b_corr / (np.sqrt(v_b_corr) + self.epsilon)

        # descent gradients
        layer.weights += -self.current_lr * grad_w
        layer.biases += -self.current_lr * grad_b

    def post_update_params(self):
        # Increment iteration counter
        self.iteration += 1



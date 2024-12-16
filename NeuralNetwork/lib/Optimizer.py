from lib.Layer import Dense
import numpy as np


class Optimizer:
    def __init__(self, lr=0.01, decay=0):
        self.lr = self.current_lr = lr
        self.iteration = 0  # Update counter
        self.decay = decay  # Decay factor

    # Adjust learning rate with decay
    def pre_update_params(self):
        self.current_lr = self.lr / (1 + self.decay * self.iteration)

    def update_params(self):
        raise NotImplementedError

    # Increment iteration counter
    def post_update_params(self):
        self.iteration += 1

    def update_network(self, layers: list[Dense]):
        # Perform updates in sequence with optional hooks
        if hasattr(self, 'pre_update_params'):
            self.pre_update_params()
        for layer in layers:
            self.update_params(layer)
        if hasattr(self, 'post_update_params'):
            self.post_update_params()


class SGD(Optimizer):
    def __init__(self, lr=1, decay=0, momentum_factor=0):
        super().__init__(lr, decay)
        self.momentum_factor = momentum_factor  # Momentum factor

    def update_params(self, layer: Dense):
        # Initialize momentum if not present
        if not hasattr(layer, 'momentum'):
            layer.momentum = {
                "weights": np.zeros_like(layer.weights),
                "biases": np.zeros_like(layer.biases)
            }

        # Compute momentum-based updates
        layer.momentum["weights"] = self.momentum_factor * layer.momentum["weights"] - self.current_lr * layer.dLoss_dWeights
        layer.momentum["biases"] = self.momentum_factor * layer.momentum["biases"] - self.current_lr * layer.dLoss_dBiases

        # Apply updates
        layer.weights += layer.momentum["weights"]
        layer.biases += layer.momentum["biases"]


class AdaGrad(Optimizer):
    def __init__(self, lr=1, decay=0, epsilon=1e-7):
        super().__init__(lr, decay)
        self.epsilon = epsilon  # Smoothing term

    def update_params(self, layer: Dense):
        # Initialize velocity if not present
        if not hasattr(layer, 'velocity'):
            layer.velocity = {
                "weights": np.zeros_like(layer.weights),
                "biases": np.zeros_like(layer.biases)
            }

        # Accumulate squared gradients
        layer.velocity["weights"] += layer.dLoss_dWeights ** 2
        layer.velocity["biases"] += layer.dLoss_dBiases ** 2

        # Apply updates with scaled gradients
        layer.weights -= self.current_lr * layer.dLoss_dWeights / np.sqrt(layer.velocity["weights"] + self.epsilon)
        layer.biases -= self.current_lr * layer.dLoss_dBiases / np.sqrt(layer.velocity["biases"] + self.epsilon)


class RMSProp(Optimizer):
    def __init__(self, lr=1, decay=0, epsilon=1e-7, beta2=0.9):
        super().__init__(lr, decay)
        self.epsilon = epsilon  # Smoothing term
        self.beta2 = beta2  # Decay rate for velocity

    def update_params(self, layer: Dense):
        # Initialize velocity if not present
        if not hasattr(layer, 'velocity'):
            layer.velocity = {
                "weights": np.zeros_like(layer.weights),
                "biases": np.zeros_like(layer.biases)
            }

        # Update velocity (EMA of squared gradients)
        layer.velocity["weights"] = self.beta2 * layer.velocity["weights"] + (1 - self.beta2) * (layer.dLoss_dWeights ** 2)
        layer.velocity["biases"] = self.beta2 * layer.velocity["biases"] + (1 - self.beta2) * (layer.dLoss_dBiases ** 2)

        # Apply updates with scaled gradients
        layer.weights -= self.current_lr * layer.dLoss_dWeights / np.sqrt(layer.velocity["weights"] + self.epsilon)
        layer.biases -= self.current_lr * layer.dLoss_dBiases / np.sqrt(layer.velocity["biases"] + self.epsilon)


class Adam(Optimizer):
    def __init__(self, lr=0.01, decay=0, epsilon=1e-7, beta1=0.9, beta2=0.999):
        super().__init__(lr, decay)
        self.epsilon = epsilon  # Smoothing term
        self.beta1 = beta1  # Momentum decay rate
        self.beta2 = beta2  # Velocity decay rate

    def update_params(self, layer: Dense):
        # Initialize momentum and velocity if not present
        if not hasattr(layer, 'momentum'):
            layer.momentum = {
                "weights": np.zeros_like(layer.weights),
                "biases": np.zeros_like(layer.biases)
            }
            layer.velocity = {
                "weights": np.zeros_like(layer.weights),
                "biases": np.zeros_like(layer.biases)
            }

        # Update momentum (EMA of gradients)
        layer.momentum["weights"] = self.beta1 * layer.momentum["weights"] + (1 - self.beta1) * layer.dLoss_dWeights
        layer.momentum["biases"] = self.beta1 * layer.momentum["biases"] + (1 - self.beta1) * layer.dLoss_dBiases

        # Update velocity (EMA of squared gradients)
        layer.velocity["weights"] = self.beta2 * layer.velocity["weights"] + (1 - self.beta2) * (layer.dLoss_dWeights ** 2)
        layer.velocity["biases"] = self.beta2 * layer.velocity["biases"] + (1 - self.beta2) * (layer.dLoss_dBiases ** 2)

        # Correct bias for momentum and velocity
        momentum_corr = {
            "weights": layer.momentum["weights"] / (1 - self.beta1 ** (self.iteration + 1)),
            "biases": layer.momentum["biases"] / (1 - self.beta1 ** (self.iteration + 1))
        }
        velocity_corr = {
            "weights": layer.velocity["weights"] / (1 - self.beta2 ** (self.iteration + 1)),
            "biases": layer.velocity["biases"] / (1 - self.beta2 ** (self.iteration + 1))
        }

        # Compute and apply updates
        layer.weights -= self.current_lr * momentum_corr["weights"] / (np.sqrt(velocity_corr["weights"]) + self.epsilon)
        layer.biases -= self.current_lr * momentum_corr["biases"] / (np.sqrt(velocity_corr["biases"]) + self.epsilon)

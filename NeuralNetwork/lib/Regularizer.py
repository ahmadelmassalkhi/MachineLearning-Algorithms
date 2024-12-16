from lib.Layer import Dense
import numpy as np


class Regularizer:
    def __init__(self, _lambda1=5e-4, _lambda2=5e-4):
        self._lambda = {
            "weights": _lambda1,
            "biases": _lambda2
        }
        self.loss = {
            "weights": 0,
            "biases": 0
        }

    def reset_loss(self):
        self.loss["weights"] = 0
        self.loss["biases"] = 0

    def get_loss(self):
        return self.loss["weights"] + self.loss["biases"]

    def add_loss(self, layer: Dense):
        raise NotImplementedError

    def backward(self, layer: Dense):
        raise NotImplementedError


class L1(Regularizer):
    def __init__(self, _lambda1=5e-7, _lambda2=5e-7):
        super().__init__(_lambda1, _lambda2)

    def add_loss(self, layer: Dense):
        self.loss["weights"] += self._lambda["weights"] * np.sum(np.abs(layer.weights))
        self.loss["biases"] += self._lambda["biases"] * np.sum(np.abs(layer.biases))

    def backward(self, layer: Dense):
        layer.dLoss_dWeights += self._lambda["weights"] * np.sign(layer.weights)
        layer.dLoss_dBiases += self._lambda["biases"] * np.sign(layer.biases)


class L2(Regularizer):
    def __init__(self, _lambda1=5e-7, _lambda2=5e-7):
        super().__init__(_lambda1, _lambda2)

    def add_loss(self, layer: Dense):
        self.loss["weights"] += self._lambda["weights"] * np.sum(layer.weights ** 2)
        self.loss["biases"] += self._lambda["biases"] * np.sum(layer.biases ** 2)

    def backward(self, layer: Dense):
        layer.dLoss_dWeights += 2 * self._lambda["weights"] * layer.weights
        layer.dLoss_dBiases += 2 * self._lambda["biases"] * layer.biases

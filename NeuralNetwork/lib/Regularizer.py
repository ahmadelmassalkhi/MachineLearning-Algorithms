from lib.Layer import Layer
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

    def add_loss(self, layer:Layer):
        raise NotImplementedError
    
    def backward(self, layer:Layer):
        raise NotImplementedError
    
    def get_loss(self):
        raise NotImplementedError
    



class L2(Regularizer):
    def __init__(self, _lambda1=5e-7, _lambda2=5e-7):
        super().__init__(_lambda1, _lambda2)

    def add_loss(self, layer:Layer):
        self.loss["weights"] += self._lambda["weights"] * np.sum(layer.weights ** 2)
        self.loss["biases"] += self._lambda["biases"] * np.sum(layer.biases ** 2)

    def backward(self, layer: Layer):
        layer.dLoss_dWeights += 2 * self._lambda["weights"] * layer.weights
        layer.dLoss_dBiases += 2 * self._lambda["biases"] * layer.biases

    def get_loss(self):
        return self.loss["weights"] + self.loss["biases"]
    
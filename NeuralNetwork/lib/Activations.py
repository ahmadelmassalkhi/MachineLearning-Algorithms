import numpy as np


class Activation:
    def forward(self, X):
        raise NotImplementedError
    def derivative(self, X):
        raise NotImplementedError
    def backward(self, dLoss_dOutput):
        raise NotImplementedError
    
    @staticmethod
    def create(name: str) -> "Activation":
        if name.lower() == 'relu': return ReLU()
        if name.lower() == 'softmax': return Softmax()
        raise ValueError(f"Activation {name} not supported.")


class ReLU(Activation):
    def forward(self, X):
        self.input = X # cache for backward pass
        return np.maximum(X, 0)

    def derivative(self, X):
        return np.where(X>0, 1, 0)

    def backward(self, dLoss_dOutput):
        return dLoss_dOutput * self.derivative(self.input)


class Softmax(Activation):
    def forward(self, X):
        exp = np.exp(X - np.max(X, axis=1, keepdims=True))
        self.output = exp / np.sum(exp, axis=1, keepdims=True) # cache for backward pass
        return self.output
    
    def backward(self, dLoss_dOutput):
        return self.output * (dLoss_dOutput - np.sum(dLoss_dOutput * self.output, axis=1, keepdims=True))
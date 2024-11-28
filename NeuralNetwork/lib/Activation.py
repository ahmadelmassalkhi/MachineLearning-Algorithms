import numpy as np

class Activation:
    def forward(self, X):
        pass
    def derivative(self, X):
        pass


class ReLU(Activation):
    def forward(self, X):
        self.inputs = X
        self.output = np.maximum(X, 0)

    def derivative(self, X):
        return np.where(X>0, 1, 0)

    def backward(self, dLoss_dValues):
        self.dLoss_dInputs = dLoss_dValues.copy()
        self.dLoss_dInputs[self.inputs <= 0] = 0
        # self.dLoss_dInputs = dLoss_dValues * self.derivative(self.inputs)


class Softmax(Activation):
    def forward(self, X):
        exp = np.exp(X - np.max(X, axis=1, keepdims=True))
        self.output = exp / np.sum(exp, axis=1, keepdims=True)
    

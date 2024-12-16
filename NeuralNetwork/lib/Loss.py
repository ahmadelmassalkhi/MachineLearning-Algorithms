import numpy as np


class Loss:
    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError


class CategoricalCrossEntropy(Loss):
    def forward(self, inputs, y_real):
        # ensure one hot encoding
        if y_real.shape != inputs.shape:
            y_real = np.eye(inputs.shape[1])[y_real]

        # compute negative log likelihood
        self.output = -np.log(np.clip(np.sum(y_real * inputs, axis=1), 1e-7, 1-1e-7))
        
        # cache
        self.y_real = y_real
        self.inputs = inputs

    def backward(self):
        self.dLoss_dInputs = - self.y_real / self.inputs


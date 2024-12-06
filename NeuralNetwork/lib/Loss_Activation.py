import numpy as np
from lib.Activation import Softmax
from lib.Loss import CategoricalCrossEntropy

class Softmax_CategoricalCrossEntropy:
    def __init__(self) -> None:
        self.softmax = Softmax()
        self.loss = CategoricalCrossEntropy()

    def forward(self, inputs, y_real=None):
        self.softmax.forward(inputs)
        if y_real is not None:
            self.y_real = y_real
            self.loss.forward(self.softmax.output, y_real)

    def backward(self):
        # make sure y_real is one hot encoded
        if self.y_real.shape != self.softmax.output.shape:
            self.y_real = np.eye(self.softmax.output.shape[1])[self.y_real]
        self.dLoss_dInputs = (self.softmax.output - self.y_real) / len(self.y_real)
        
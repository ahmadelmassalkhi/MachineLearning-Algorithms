import numpy as np


class Loss:
    def forward(self, input, y_real):
        raise NotImplementedError
    def backward(self):
        raise NotImplementedError

    @staticmethod
    def create(name: str) -> "Loss":
        if name.lower() == 'categoricalcrossentropy': return CategoricalCrossEntropy()
        raise ValueError(f"Loss function {name} not supported.")

class CategoricalCrossEntropy(Loss):
    def forward(self, input, y_real):
        # ensure one hot encoding
        if y_real.shape != input.shape:
            y_real = np.eye(input.shape[1])[y_real]
        
        # cache for backward pass
        self.y_real = y_real
        self.input = input

        # compute & return negative log likelihood
        return -np.log(np.clip(np.sum(y_real * input, axis=1), 1e-7, 1-1e-7))

    def backward(self):
        # compute & return dLoss_dInput
        return - self.y_real / np.clip(self.input, 1e-7, None) # prevents exploding outputs (due to unnormalized/large inputs)

import numpy as np

class Loss:
    def forward(self):
        pass
    def backward(self):
        pass


class CategoricalCrossEntropy(Loss):
    def forward(self, inputs, y_real):
        if y_real.shape[-1] != inputs.shape[0]: 
            raise ValueError(f'Incompatible shapes of predicted:real as {inputs.shape}:{y_real.shape}')

        # incase of one hot encoding
        if y_real.shape != inputs.shape:
            y_real = np.eye(inputs.shape[1])[y_real]

        # compute negative log likelihood
        self.output = -np.log(np.clip(np.sum(y_real * inputs, axis=1), 1e-7, 1-1e-7))
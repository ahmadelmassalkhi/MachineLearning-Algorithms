import numpy as np

class Loss:
    def forward(self):
        pass
    def backward(self):
        pass


class CategoricalCrossEntropy(Loss):
    def forward(self, inputs:np.array, y_real:np.array):
        k = len(inputs)

        # incase of one hot encoding
        if y_real.shape == inputs.shape:
            correct_confidences = np.sum(y_real * inputs, axis=1)
                
        # incase of one list
        elif len(y_real.shape)==1 and y_real.shape[0] == k:
            correct_confidences = inputs[range(k), y_real]

        else: raise ValueError(f'Incompatible shapes of predicted:real as {inputs.shape}:{y_real.shape}')

        # compute negative log likelihood
        self.output = -np.log(np.clip(correct_confidences, 1e-7, 1-1e-7))
import numpy as np


class Loss:
    def forward(self, prediction, real):
        raise NotImplementedError
    def backward(self):
        raise NotImplementedError

    @staticmethod
    def create(name: str) -> "Loss":
        if name.lower() == 'categorical_crossentropy': return CategoricalCrossEntropy()
        raise ValueError(f"Loss function {name} not supported.")


class CategoricalCrossEntropy(Loss):
    def forward(self, prediction, real):
        # ensure one hot encoding
        if real.shape != prediction.shape:
            real = np.eye(prediction.shape[1])[real]
        
        # cache for backward pass
        self.real = real
        self.prediction = np.clip(prediction, 1e-7, 1-1e-7)

        # compute & return negative log likelihood
        return np.mean(- real * np.log(self.prediction))
        
    def backward(self):
        # compute & return dLoss_dprediction
        return - self.real / self.prediction # prevents exploding outputs (due to unnormalized/large inputs)


class BinaryCrossEntropy(Loss):
    def forward(self, prediction, real):
        # Cache for backward pass
        self.real = real
        self.prediction = np.clip(prediction, 1e-7, 1 - 1e-7)

        # Compute and return the negative log likelihood
        return np.mean(-real * np.log(self.prediction) - (1 - real) * np.log(1 - self.prediction))

    def backward(self):
        # Compute gradient dLoss/dPrediction
        return - (self.real / self.prediction) + (1 - self.real) / (1 - self.prediction)

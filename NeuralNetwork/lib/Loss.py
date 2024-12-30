import numpy as np


class Loss:
    def forward(self, prediction:np.ndarray, real:np.ndarray):
        raise NotImplementedError
    def backward(self):
        raise NotImplementedError

    @staticmethod
    def create(name: str) -> "Loss":
        if name.lower() == 'categorical_crossentropy': return CategoricalCrossEntropy()
        raise ValueError(f"Loss function {name} not supported.")


class CategoricalCrossEntropy(Loss):
    def forward(self, prediction:np.ndarray, real:np.ndarray):
        # ensure one hot encoding
        try:
            if real.shape != prediction.shape:
                real = np.eye(prediction.shape[1])[real]
        except Exception as e:
            print(e)
            raise ValueError(f"Incompatible shapes: {real.shape} and {prediction.shape}.")
        
        # cache for backward pass
        self.real = real
        self.prediction = np.clip(prediction, 1e-7, 1-1e-7) # prevents exploding outputs (due to unnormalized/large inputs)

        # compute & return negative log likelihood
        return np.mean(- real * np.log(self.prediction))
        
    def backward(self):
        # compute & return dLoss_dprediction
        return - self.real / self.prediction 

class BinaryCrossEntropy(Loss):
    def forward(self, prediction:np.ndarray, real:np.ndarray):
        # Ensure shapes match for element-wise operations
        if real.shape != prediction.shape:
            raise ValueError(f"Incompatible shapes: {real.shape} and {prediction.shape}.")

        # Cache for backward pass
        self.real = real
        self.prediction = np.clip(prediction, 1e-7, 1 - 1e-7)

        # Compute and return the negative log likelihood
        return np.mean(-real * np.log(self.prediction) - (1 - real) * np.log(1 - self.prediction))

    def backward(self):
        # Compute gradient dLoss/dPrediction
        return - (self.real / self.prediction) + (1 - self.real) / (1 - self.prediction)


class MeanSquaredError(Loss):
    def forward(self, prediction:np.ndarray, real:np.ndarray):
        # Ensure shapes match for element-wise operations
        if real.shape != prediction.shape:
            raise ValueError(f"Incompatible shapes: {real.shape} and {prediction.shape}.")

        # Cache for backward pass
        self.real = real
        self.prediction = prediction

        # Compute and return MSE loss
        return np.mean(np.square(prediction - real))

    def backward(self):
        # Gradient of MSE: 2 * (prediction - real) / N
        n = self.real.shape[0]
        return (2 / n) * (self.prediction - self.real)
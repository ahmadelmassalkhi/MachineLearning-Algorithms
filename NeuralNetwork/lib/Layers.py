import numpy as np


class Layer:
    def forward(self, input:np.ndarray):
        raise NotImplementedError

    def backward(self, dLoss_dOutput:np.ndarray):
        raise NotImplementedError


class Dense(Layer):
    def __init__(self, nbOfOutputs: int, activation:str=None, regularizer:str=None) -> None:
        # init biases
        self.nbOfOutputs = nbOfOutputs
        self.biases = np.zeros(nbOfOutputs)
        
        # init activation function & regularizer
        from lib.Activations import Activation
        from lib.Regularizers import Regularizer
        self.activation = Activation.create(name=activation) if activation else None
        self.regularizer = Regularizer.create(name=regularizer) if regularizer else None


    def _init_weights(self, input:np.ndarray):
        self.nbOfInputs = input.shape[-1]
        self.weights = np.random.randn(self.nbOfInputs, self.nbOfOutputs) * np.sqrt(2 / self.nbOfInputs)


    def forward(self, X:np.ndarray):
        # init weights if not done yet
        if not hasattr(self, 'weights'): self._init_weights(X)

        # cache for backward pass
        self.input = X

        # add regularization loss if exists
        if self.regularizer: 
            self.regularizer.add_loss(self)
        
        # compute & return output
        return self.activation.forward(np.dot(X, self.weights) + self.biases) if self.activation else np.dot(X, self.weights) + self.biases


    def backward(self, dLoss_dActivation:np.ndarray):
        # compute gradients
        dLoss_dNeurons = self.activation.backward(dLoss_dActivation) if self.activation else dLoss_dActivation
        self.dLoss_dWeights = np.dot(self.input.T, dLoss_dNeurons)
        self.dLoss_dBiases = np.average(dLoss_dNeurons, axis=0) # condense to 1xm dimensions
        
        # regularize parameters
        if self.regularizer:
            self.regularizer.backward(self)
        
        # compute & return dLoss_dInput
        return np.dot(dLoss_dNeurons, self.weights.T)



class Dropout(Layer):
    def __init__(self, rate: float):
        self.initial_rate = self.rate = rate
        self.set_training_mode(True)

    def set_training_mode(self, training:bool):
        self.training_mode = training

    def forward(self, input:np.ndarray):
        if not self.training_mode: return input
        self.binomial = np.random.binomial(1, 1-self.rate, input.shape) / (1 - self.rate)
        return input * self.binomial

    def backward(self, dLoss_dOutput:np.ndarray):
        if not self.training_mode: return dLoss_dOutput
        return dLoss_dOutput * self.binomial



class Flatten(Layer):
    def forward(self, input:np.ndarray):
        if len(input.shape) > 2: 
            return np.reshape(input, (input.shape[0], -1))  # Flatten only if input is multi-dimensional
        return input
    
    def backward(self, dLoss_dOutput:np.ndarray):
        return None
from lib.Layer import Layer




class Optimizer_GD:
    def __init__(self, lr=1) -> None:
        self.lr = lr

    def update_params(self, layer:Layer):
        layer.weights -= self.lr * layer.dLoss_dWeights
        layer.biases -= self.lr * layer.dLoss_dBiases
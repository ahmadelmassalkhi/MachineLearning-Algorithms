import nnfs
import nnfs.datasets
import pickle
from lib.Layer import Layer
from lib.Activation import *
from lib.Loss_Activation import *
from lib.Optimizer import *
from lib.Regularizer import *


class model:
    def __init__(self, nbOfInputs: int, nbOfOutputs: int, with_regularization: bool = False):
        self.nbOfInputs = nbOfInputs
        self.nbOfOutputs = nbOfOutputs

        # init activation functions
        self.activation1 = ReLU()
        self.activation2 = ReLU()
        self.activationN = Softmax()
        self.Softmax_Loss = Softmax_CategoricalCrossEntropy()

        # init optimizer
        self.optimizer = Adam()
        if with_regularization:
            self.regularizer = L2(1e-6, 0)
        else:
            self.regularizer = 0

    def load(self, model_path: str):
        # get model
        try:
            # load model
            self.model_path = model_path
            with open(model_path, 'rb') as file:
                self.layer1, self.layer2, self.layerN = pickle.load(file)
            print(f'Model loaded from {model_path}')
        except FileNotFoundError:
            # init model
            self.layer1 = Dense(self.nbOfInputs, 256)
            self.layer2 = Dense(self.layer1.nbOfOutputs, 128)
            self.layerN = Dense(self.layer2.nbOfOutputs, self.nbOfOutputs)
            print(f"Model `{model_path}` initialized from scratch")

    def save(self):
        # Save layers to a file (to save their weights & biases)
        with open(self.model_path, 'wb') as file:
            pickle.dump([self.layer1, self.layer2, self.layerN], file)
            print(f'Model saved to {self.model_path}')

    def forward(self, X, y=None):
        self.layer1.forward(X)
        self.activation1.forward(self.layer1.output)

        self.layer2.forward(self.activation1.output)
        self.activation2.forward(self.layer2.output)

        self.layerN.forward(self.activation2.output)
        if y is not None:
            ''' EVALUATE MODEL PREDICTION '''
            self.Softmax_Loss.forward(self.layerN.output, y)
            self.loss = np.average(self.Softmax_Loss.loss.output)
            self.accuracy = np.mean(
                np.argmax(self.Softmax_Loss.softmax.output, axis=1) == y)
        else:
            self.activationN.forward(self.layerN.output)

    # performs full propagation until reaching a target-confidence
    ''' X: numpy array of shape (any, nbOfInputs) '''

    def learn(self, X, y: list[int], target_accuracy: float = 0.99):
        if len(X) != len(y):
            raise ValueError(f'Number of Features != Labels!')
        # propagate
        self.accuracy = i = 0
        while self.accuracy < target_accuracy:
            self.forward(X, y)

            ''' LOG MODEL EVALUATION '''
            if not i % 100:
                print(f'epoch {i}, ' +
                      f'acc: {self.accuracy:.3f}, ' +
                      f'loss: {self.loss:.3f}, ' +
                      f'lr: {self.optimizer.current_lr}')

            # backward propagation
            self.Softmax_Loss.backward()
            self.layerN.backward(self.Softmax_Loss.dLoss_dInputs)
            if self.regularizer:
                self.regularizer.backward(self.layerN)

            self.activation2.backward(self.layerN.dLoss_dInputs)
            self.layer2.backward(self.activation2.dLoss_dInputs)
            if self.regularizer:
                self.regularizer.backward(self.layer2)

            self.activation1.backward(self.layer2.dLoss_dInputs)
            self.layer1.backward(self.activation1.dLoss_dInputs)
            if self.regularizer:
                self.regularizer.backward(self.layer1)

            # optimize / descent gradient
            self.optimizer.pre_update_params()
            self.optimizer.update_params(self.layer1)
            self.optimizer.update_params(self.layer2)
            self.optimizer.update_params(self.layerN)
            self.optimizer.post_update_params()
            i += 1


nnfs.init()
n_samples, n_classes = 1000, 3
(X_train, y_train) = nnfs.datasets.spiral_data(n_samples, n_classes)

# init & train model
_model = model(nbOfInputs=2, nbOfOutputs=n_classes, with_regularization=True)
_model.load(model_path='model.pkl')
_model.learn(X_train, y_train, 0.9)


def evaluate(_model: model, X_test, y_test):
    _model.forward(X_test, y_test)
    print(f'acc: {_model.accuracy:.3f}, ' +
          f'loss: {_model.loss:.3f}, ' +
          f'lr: {_model.optimizer.current_lr}')


(X_test, y_test) = nnfs.datasets.spiral_data(100, n_classes)
evaluate(_model, X_test, y_test)

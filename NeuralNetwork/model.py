import nnfs
import nnfs.datasets
import pickle
from lib.Layer import *
from lib.Activation import *
from lib.Loss_Activation import *
from lib.Optimizer import *
from lib.Regularizer import *


class model:
    def __init__(self, nbOfInputs: int, nbOfOutputs: int, regularize: bool = False, dropout_rate: float = 0.0):
        self.nbOfInputs = nbOfInputs
        self.nbOfOutputs = nbOfOutputs

        # init activation functions
        self.activation1 = ReLU()
        self.activation2 = ReLU()
        self.activationN = Softmax()

        self.dropout1 = Dropout(rate=dropout_rate)
        self.dropout2 = Dropout(rate=dropout_rate)
        self.Softmax_Loss = Softmax_CategoricalCrossEntropy()

        # init optimizer
        self.optimizer = Adam()
        if regularize:
            self.regularizer = L2(1e-6, 0)
        else:
            self.regularizer = None

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
        self.dropout1.forward(self.activation1.output)

        self.layer2.forward(self.dropout1.output)
        self.activation2.forward(self.layer2.output)
        self.dropout2.forward(self.activation2.output)

        self.layerN.forward(self.dropout2.output)
        if y is not None:
            ''' EVALUATE MODEL PREDICTION '''
            self.Softmax_Loss.forward(self.layerN.output, y)
            self.loss = np.average(self.Softmax_Loss.loss.output)
            self.accuracy = np.mean(
                np.argmax(self.Softmax_Loss.softmax.output, axis=1) == y)
        else:
            self.activationN.forward(self.layerN.output)

    # performs full propagation until reaching a target-accuracy
    ''' X: numpy array of shape (any, nbOfInputs) '''

    def learn(self, X, y: list[int], target_accuracy: float = 0.99, iterations_limit: int = 10001):
        if len(X) != len(y):
            raise ValueError(f'Number of Features != Labels!')
        # propagate
        self.accuracy = i = 0
        while self.accuracy < target_accuracy and i < iterations_limit:
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

            self.dropout2.backward(self.layerN.dLoss_dInputs)
            self.activation2.backward(self.dropout2.dLoss_dInputs)
            self.layer2.backward(self.activation2.dLoss_dInputs)

            self.dropout1.backward(self.layer2.dLoss_dInputs)
            self.activation1.backward(self.dropout1.dLoss_dInputs)
            self.layer1.backward(self.activation1.dLoss_dInputs)

            if self.regularizer:
                self.regularizer.backward(self.layerN)
                self.regularizer.backward(self.layer2)
                self.regularizer.backward(self.layer1)

            # optimize / descent gradient
            self.optimizer.pre_update_params()
            self.optimizer.update_params(self.layer1)
            self.optimizer.update_params(self.layer2)
            self.optimizer.update_params(self.layerN)
            self.optimizer.post_update_params()
            i += 1


nnfs.init()
n_samples, n_classes = 100, 3
(X_train, y_train) = nnfs.datasets.spiral_data(n_samples, n_classes)

# init & train model
# _model = model(nbOfInputs=2, nbOfOutputs=n_classes)
_model = model(nbOfInputs=2, nbOfOutputs=n_classes,
               regularize=True, dropout_rate=0.5)
_model.load(model_path='model.pkl')
_model.learn(X_train, y_train, 0.85)


def evaluate(_model: model, X_test, y_test):
    _model.forward(X_test, y_test)
    print(f'acc: {_model.accuracy:.3f}, ' +
          f'loss: {_model.loss:.3f}, ' +
          f'lr: {_model.optimizer.current_lr}')


(X_test, y_test) = nnfs.datasets.spiral_data(100000, n_classes)
evaluate(_model, X_test, y_test)

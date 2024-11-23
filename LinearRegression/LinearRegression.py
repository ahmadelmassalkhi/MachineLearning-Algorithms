import numpy as np

class LinearRegression:
    def __init__(self, lr=0.01, n_iters=10000) -> None:
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features) # shape = (n_features,)
        self.bias = 0

        for _ in range(self.n_iters):
            # predict all y values
            y_predicted = np.dot(X, self.weights) + self.bias #produces n_samples of y predictions
            
            # derivatives with respect to w & b
            dw = (2/n_samples) * np.dot(X.T, y_predicted - y)
            db = (2/n_samples) * np.sum(y_predicted - y)
            
            # update (gradient descent method)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
        return self

    def predict(self, X_test):
        return np.dot(X_test, self.weights) + self.bias
    


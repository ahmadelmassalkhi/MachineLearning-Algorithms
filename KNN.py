import numpy as np
from collections import Counter


# Function to calculate Euclidean distance
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

class KNN:
    def __init__(self, k=3) -> None:
        self.k = k

    def fit(self, X_train, Y_train):
        self.X_train = X_train
        self.y_train = y_train
        return self

    def predict(self, X_test):
        return np.array([self._predict(x) for x in X_test])

    def _predict(self, x:list):
        # measure distances
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_labels).most_common(1) # [(element, frequency)]
        return most_common[0][0]


# Sample data (X_train: features, y_train: labels)
X_train = np.array([[1, 2], [2, 3], [3, 4], [6, 8], [7, 8], [8, 9]])
y_train = np.array([0, 0, 0, 1, 1, 1])

# Test data point
X_test = np.array([[5, 5]])

prediction = KNN(k=3).fit(X_train, y_train).predict(X_test)[0]
print(f'Prediction = {prediction}')
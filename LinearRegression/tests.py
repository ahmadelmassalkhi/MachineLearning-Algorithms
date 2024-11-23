import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt


''' GENERATE TRAINING & TEST DATASET '''
X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X, -y, test_size=0.2, random_state=1234)


''' EVALUATE ACCURACY '''
def rmse(y_predicted, y_true): # root of mean squared error
    return np.sqrt(np.mean((y_true - y_predicted)**2))


''' DRAW TRAINING/TEST/PREDICTIONS '''
def plot(regressor):
    plt.figure(figsize=(8,6)) # 8x6 inches figure
    cmap = plt.get_cmap('viridis') # color map
    plt.scatter(X_train, y_train, color=cmap(0.9), s=10) # draw the training data (x,y values)
    plt.scatter(X_test, y_test, color=cmap(0.5), s=10) # draw the test data with their REAL values (x_test -> y_test real values)
    plt.plot(X, regressor.predict(X), color='black', linewidth=2, label="Prediction") # draw the line passing through predictions of all data (test & training)
    plt.show()

# `scatter` plots the points
# `plot` plots & connects the points into a line


from LinearRegression import LinearRegression
plot(LinearRegression().fit(X_train, y_train))
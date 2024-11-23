import numpy as np
np.random.seed(0)


n_outputs = 3
inputs = np.array([1,2,3,4])
weights = np.random.randn(len(inputs), n_outputs)
biases = np.zeros(n_outputs)


def relu(x):
    return np.maximum(x,0)

def relu_derivative(x):
    return np.where(x>0, 1,0)


def backward(inputs, expected_output=0, lr=0.001, n_iterations=100):
    global weights, biases
    for i in range(n_iterations):
        # compute loss
        Linear = np.dot(inputs, weights) + biases
        ReLU = relu(Linear)
        output = np.sum(ReLU)
        loss = (output - expected_output) ** 2
        print(f'Iteration {i+1} Loss = {loss}')

        # compute gradients
        dLoss_dLinear = 2 * (output - expected_output) * relu_derivative(Linear)
        dLoss_dWeights = np.outer(inputs, dLoss_dLinear)
        dLoss_dBiases = dLoss_dLinear

        # descent gradients
        weights -= lr * dLoss_dWeights
        biases -= lr * dLoss_dBiases
    # print stats
    print()
    print(f'Final Output = {output}')
    print(f'Final Weights = {weights}')
    print(f'Final Biases = {biases}')


backward(inputs)
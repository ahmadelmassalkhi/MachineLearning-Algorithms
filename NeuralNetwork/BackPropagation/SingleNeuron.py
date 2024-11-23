import numpy as np
np.random.seed(0) # refer to NOTE


'''
NOTE:
    consequences of a single-neuron network
    * ReLU(-ve) is a killer of this single-neuron network 
    * because all gradients converge to 0 => no learning, dead network

    to prevent this
    * pick a seed such that forward pass does NOT result in a (-ve or 0) output    
    * pick a targetOutput >= 0
'''

# input neurons from dataset
inputs = np.array([5,2,1])
targetOutput = 5 # must be >= 0

# initialize weights of the SINGLE neuron, randomly 
weights = np.random.randn(len(inputs))
bias = 0


def relu(x):
    return np.maximum(x,0)

def relu_derivative(x):
    return np.where(x>0, 1,0)

def backward(lr=0.001, n_iterations=100):
    global weights, bias
    for i in range(n_iterations):
        # calculate loss & essentials
        linearOutput = np.dot(inputs, weights) + bias
        output = relu(linearOutput)
        loss = (output - targetOutput) ** 2
        print(f'Iteration {i+1}')
        print(f'Output = {output}')
        print(f'Loss = {loss}')

        # compute gradients
        dLoss_dLinear = 2 * (output - targetOutput) * relu_derivative(linearOutput)
        dLoss_dWeights = dLoss_dLinear * inputs
        dLoss_dBias = dLoss_dLinear * 1

        # descent gradients
        weights -= lr * dLoss_dWeights
        bias -= lr * dLoss_dBias

        # print stats
        print(f'Weights after iteration: {weights}')
        print(f'Bias after iteration: {bias}')
        print()

backward()
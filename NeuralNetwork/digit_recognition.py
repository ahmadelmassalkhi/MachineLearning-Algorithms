import pickle
from lib.Layer import Layer
from lib.Activation import ReLU, Softmax
from lib.Loss_Activation import Softmax_CategoricalCrossEntropy
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from lib.Optimizer import *



class NeuralNetwork:
    def __init__(self, model_path:str):
        self.image_dims = (self.image_width, self.image_height) = (28, 28)
        self.nbOfInputs, self.nbOfOutputs = self.image_width * self.image_height, 10

        # get model
        try:
            # load model
            self.model_path = model_path
            with open(model_path, 'rb') as file:
                self.layer1, self.layer2, self.layerN = pickle.load(file)
            print(f'Model loaded from {model_path}')
        except FileNotFoundError:
            # init model
            self.layer1 = Layer(self.nbOfInputs, 256)
            self.layer2 = Layer(self.layer1.nbOfOutputs, 128)
            self.layerN = Layer(self.layer2.nbOfOutputs, self.nbOfOutputs)
            print(f"Model `{model_path}` initialized from scratch")

        # init activation functions
        self.activation1 = ReLU()
        self.activation2 = ReLU()
        self.activationN = Softmax()
        self.Softmax_Loss = Softmax_CategoricalCrossEntropy()

        # init optimizer
        self.optimizer = SGD(lr=0.1, decay=1e-7, momentum_factor=0.99)
        # self.optimizer = Adam()

    ''' X: numpy array of shape (any, 28x28) '''
    def forward(self, X):
        # propagate
        self.layer1.forward(X)
        self.activation1.forward(self.layer1.output)

        self.layer2.forward(self.activation1.output)
        self.activation2.forward(self.layer2.output)

        self.layerN.forward(self.activation2.output)
        self.activationN.forward(self.layerN.output)
        
        self.output_confidence = np.max(self.activationN.output, axis=1)
        self.output = np.argmax(self.activationN.output, axis=1)
        return self.output

    # performs full propagation until reaching a target-confidence
    ''' images: numpy array of shape (any, 28x28) '''
    def learn(self, images, labels:list[int], target_confidence:float=0.99):
        if len(images)!=len(labels):
            raise ValueError(f'Number of Features != Labels!')
        # propagate
        confidence = i = 0
        while confidence < target_confidence:
            self.forward(images)
            self.Softmax_Loss.forward(self.layerN.output, np.array(labels))

            ''' EVALUATE MODEL PREDICTION '''
            loss = np.average(self.Softmax_Loss.loss.output)
            accuracy = np.mean(np.argmax(self.Softmax_Loss.softmax.output, axis=1) == labels)
            if not i % 100:
                print(f'epoch {i}, '+
                    f'acc: {accuracy:.3f}, ' +
                    f'loss: {loss:.3f}, ' +
                    f'lr: {self.optimizer.current_lr}')

            # backward propagation
            self.Softmax_Loss.backward()
            self.layerN.backward(self.Softmax_Loss.dLoss_dInputs)

            self.activation2.backward(self.layerN.dLoss_dInputs)
            self.layer2.backward(self.activation2.dLoss_dInputs)

            self.activation1.backward(self.layer2.dLoss_dInputs)
            self.layer1.backward(self.activation1.dLoss_dInputs)

            # optimize / descent gradient
            self.optimizer.pre_update_params()
            self.optimizer.update_params(self.layer1)
            self.optimizer.update_params(self.layer2)
            self.optimizer.update_params(self.layerN)
            self.optimizer.post_update_params()

            # print stats
            confidence = np.average(self.Softmax_Loss.softmax.output[range(len(labels)), labels])
            print(f'Iteration {i+1}: Average Correct Confidence = {confidence}')
            i+=1

    def save(self):
        # Save layers to a file (to save their weights & biases)
        with open(self.model_path, 'wb') as file:
            pickle.dump([self.layer1, self.layer2, self.layerN], file)
            print(f'Model saved to {self.model_path}')


    def matricize_image(self, img: Image.Image):
        # Crop the image to square
        width, height = img.size
        if width > height:
            img = img.crop(((width - height) // 2, 0, (width + height) // 2, height))
        else:
            img = img.crop((0, (height - width) // 2, width, (height + width) // 2))

        # Resize the image to the target dimensions
        resized_image = img.resize((self.image_width, self.image_height))

        # Convert to grayscale numpy array and normalize
        return np.array(resized_image.convert('L')).astype('float32') / 255.0


    def matricize_images(self, images:list[Image.Image]):
        for i in range(len(images)):
            images[i] = self.matricize_image(images[i])
        if len(images)==0: return np.empty((0, self.image_width, self.image_height))
        return np.array(images)
    

    def plot_image(self, image_matrix:np.ndarray, prediction:int):
        # Plot the first image in the test set and its predicted label
        plt.imshow(image_matrix, cmap='gray')
        plt.title(f"Predicted: {prediction}")
        plt.show()
        
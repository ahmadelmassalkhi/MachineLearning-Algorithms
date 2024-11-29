import os
import glob
from PIL import Image
import matplotlib.pyplot as plt
from lib.neural_network import NeuralNetwork
# from tensorflow.keras.datasets import mnist # type: ignore


def get_nth_png():
    try:
        return Image.open(glob.glob(os.path.join(os.getcwd(), '*.png'))[0])
    except Exception as e:
        print(e)
        exit()

def plot_image(image_matrix, prediction):
    # Plot the first image in the test set and its predicted label
    plt.imshow(image_matrix, cmap=plt.cm.binary)
    plt.title(f"Predicted: {prediction}")
    plt.show()


# (X_train, y_train), (X_test, y_test) = mnist_dataset()

# load networks
nn = NeuralNetwork(model_path='model.pkl')
image = nn.prepare_image(get_nth_png())
nn.plot_image(image, nn.forward(image)[0])
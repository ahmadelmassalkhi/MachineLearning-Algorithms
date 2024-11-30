import numpy as np
from PIL import Image
from pathlib import Path
# from tensorflow.keras.datasets import mnist # type: ignore
from lib.digit_recognition import NeuralNetwork
####################################################################
''' PREPARE DATASET '''


try:
    first_png = Image.open(next(Path().glob('*.png'), None))
except Exception as e:
    print(e)
    exit()


nn = NeuralNetwork()
X_train = nn.matricize_image(first_png)
nn.forward(X_train.reshape(1, 28*28))
nn.plot_image(X_train[0], nn.output[0])


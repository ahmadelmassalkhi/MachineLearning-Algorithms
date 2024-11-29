import os
import glob
import numpy as np
from PIL import Image
from lib.neural_network import NeuralNetwork
# from tensorflow.keras.datasets import mnist # type: ignore

##################################################################
''' LOAD TRAINING DATASET '''


def get_all_files(dirPath: str = './', extension:str = 'png'):
    """Returns a list of all .png files in the specified directory."""
    try:
        return glob.glob(os.path.join(dirPath, f'*.{extension}'))
    except Exception as e:
        print(e)
        exit()


all_images = []
all_labels = []
for i in range(10):
    images = [Image.open(file) for file in get_all_files(f'./Images/{i}')]
    all_images.extend(images)
    all_labels.extend([i] * len(images))


nn = NeuralNetwork(model_path='model.pkl')
nn.learn(all_images, all_labels)
nn.save()

##################################################################


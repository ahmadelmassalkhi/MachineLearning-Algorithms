from pathlib import Path
from PIL import Image
from lib.neural_network import NeuralNetwork
from tensorflow.keras.datasets import mnist # type: ignore

##################################################################
''' LOAD TRAINING DATASET '''

# (x_train, y_train), (x_test, y_test) = mnist.load_data()
def get_all_files(dir_path: str = './', extension: str = 'png') -> list:
    """Returns a list of all files with the specified extension in the given directory."""
    return list(Path(dir_path).glob(f'*.{extension}'))

all_images = []
all_labels = []
for i in range(10):
    images = [Image.open(file) for file in get_all_files(f'./Images/{i}')]
    all_images.extend(images)
    all_labels.extend([i] * len(images))

nn = NeuralNetwork(model_path='model.pkl')
all_images = nn.prepare_images(all_images).reshape(len(all_images), 28*28)
nn.learn(all_images, all_labels)
nn.save()

##################################################################


from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from lib.neural_network import NeuralNetwork
# from tensorflow.keras.datasets import mnist # type: ignore


def get_all_files(dir_path: str = './', extension: str = 'png') -> list:
    """Returns a list of all files with the specified extension in the given directory."""
    return list(Path(dir_path).glob(f'*.{extension}'))
def get_png():
    try:
        return Image.open(get_all_files()[0])
    except Exception as e:
        print(e)
        exit()

# (X_train, y_train), (X_test, y_test) = mnist_dataset()
images = [Image.open(file) for file in get_all_files()]

# load network
nn = NeuralNetwork(model_path='model.pkl')
images = nn.prepare_images(images)
flattened_images = images.reshape(len(images), 28*28)
nn.plot_image(images[0], nn.forward(flattened_images)[0])
print(f'Confidence = {nn.output_confidence[0]}')
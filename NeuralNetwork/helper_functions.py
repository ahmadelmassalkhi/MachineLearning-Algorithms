import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist # type: ignore
import glob


def mnist_dataset(nbOfBatches:int=100):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # choose batch size
    X_train = X_train[-nbOfBatches::] # (60000, 28, 28) => (batches, 28, 28)
    y_train = y_train[-nbOfBatches:] # (60000,) => (batches,)

    # Normalize pixel values to [0, 1] for better performance
    X_train = X_train.astype('float32') / 255.0

    return (X_train, y_train), (X_test, y_test)


def resize_img(image_path:str, target_height:int=28, target_width:int=28):
    # Load the image
    try:
        img = Image.open(image_path)
    except FileNotFoundError as e:
        print(e)
        exit()

    # If the image is already has target dimensions, do nothing
    if img.size == (target_height, target_width):
        return img

    # Get the current dimensions of the image
    width, height = img.size

    # Otherwise, resize and crop the image
    if width > height:
        # Crop the left/right side and resize the image
        left = (width - height) // 2
        right = left + height
        img = img.crop((left, 0, right, height))
    else:
        # Crop the top/bottom side and resize the image
        top = (height - width) // 2
        bottom = top + width
        img = img.crop((0, top, width, bottom))

    img = img.resize((target_height, target_width))
    # img.save(image_path)
    return img


def img_to_grayscale_matrix(image_matrix):
    # Convert to grayscale numpy array and normalize
    return np.array(image_matrix.convert('L')).astype('float32') / 255.0


def plot_image(image_matrix, prediction):
    # Plot the first image in the test set and its predicted label
    plt.imshow(image_matrix, cmap=plt.cm.binary)
    plt.title(f"Predicted: {prediction}")
    plt.show()

def predict_all_images(digit_folder):
    test_images = np.empty((0, 28, 28))  # Start with an empty array
    test_labels = np.empty((0,), dtype=int)  # Start with an empty array
    folder = f'./Images/{digit_folder}'
    for file in os.listdir(folder):
        if file.endswith(".png"):
            # Append the new image and label
            test_images = np.append(test_images, [img_to_grayscale_matrix(resize_img(f'{folder}/{file}'))], axis=0)
            test_labels = np.append(test_labels, [digit_folder], axis=0)
    return test_images, test_labels

def predict_png_image(file=None):
    if file == None:
        try:
            file = glob.glob(os.path.join(os.getcwd(), '*.png'))[0]
        except Exception as e:
            print(e)
            exit()
    return img_to_grayscale_matrix(resize_img(file)).reshape(1, 28, 28)



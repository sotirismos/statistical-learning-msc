import os
import pickle

import matplotlib.pyplot as plt
import numpy as np


def unpickle_file(file: str) -> dict:
    """
    Load a pickle file and return its contents as a dictionary.

    Parameters
    ----------
    file : str
        The path to the pickle file to be loaded.

    Returns
    -------
    dict
        The dictionary loaded from the pickle file.
    """
    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict


def load_cifar10_batch(file_path: str) -> dict:
    """
    Load and decode a single CIFAR-10 batch file.

    This function reads a batch file from the CIFAR-10 dataset, decodes the binary keys
    and values into UTF-8 strings where applicable, and returns a dictionary containing
    the batch data.

    Parameters
    ----------
    file_path : str
        The path to the CIFAR-10 batch file.

    Returns
    -------
    dict
        A dictionary containing keys and values from the CIFAR-10 batch file,
        with binary data decoded into strings where applicable.
    """
    with open(file_path, "rb") as f:
        batch = pickle.load(f, encoding="bytes")
        batch_decoded = {}
        for k, v in batch.items():
            key = k.decode("utf-8") if isinstance(k, bytes) else k
            if isinstance(v, bytes):
                value = v.decode("utf-8")
            elif isinstance(v, list):
                value = [
                    item.decode("utf-8") if isinstance(item, bytes) else item
                    for item in v
                ]
            else:
                value = v
            batch_decoded[key] = value
        return batch_decoded


def preprocess_images(images):
    """
    Preprocess an array of images from the CIFAR-10 dataset.

    This function reshapes the images from a flat array to the format (num_images, 32, 32, 3),
    and normalizes their pixel values to the range [0, 1].

    Parameters
    ----------
    images : np.ndarray
        A NumPy array of shape (num_images, 3 * 32 * 32) containing image data in a flattened format.

    Returns
    -------
    np.ndarray
        A NumPy array of shape (num_images, 32, 32, 3) with normalized pixel values.
    """
    # Number of images
    num_images = images.shape[0]

    # Reshape images to (num_images, 3, 32, 32)
    images_reshaped = images.reshape(num_images, 3, 32, 32)

    # Transpose to (num_images, 32, 32, 3)
    images_transposed = images_reshaped.transpose(0, 2, 3, 1)

    # Normalize the images (TODO: explain why we normalize)
    images_normalized = images_transposed / 255.0

    return images_normalized


def normalize_data(x):
    """
    Normalize the input data to have pixel values in the range [0, 1].

    This function converts the input data type to float32 and normalizes
    pixel values by dividing by 255.

    Parameters
    ----------
    x : np.ndarray
        The input array of image data, typically with values in [0, 255].

    Returns
    -------
    np.ndarray
        The normalized data as float32 with values in the range [0, 1].
    """
    # TODO: Explain why we normalize the data
    # The data is converted to float32 to ensure that the division operation results in floating-point values, preserving decimal precision.
    x_normalized = x.astype("float32") / 255.0
    return x_normalized


def display_images(images, labels, class_names, num_images):
    """
    Display a grid of images with their corresponding labels.

    Parameters
    ----------
    images : np.ndarray
        A NumPy array of images to display. Should be of shape (num_images, height, width, channels).
    labels : np.ndarray
        A NumPy array of labels corresponding to each image.
    class_names : list
        A list of class names, where class_names[label] gives the class name for that label.
    num_images : int
        The number of images to display in the grid.

    Returns
    -------
    None
        Displays a figure with a 3x3 grid (or fewer if num_images < 9) of images and their titles.
    """
    plt.figure(figsize=(10, 10))
    for i in range(num_images):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        plt.title(class_names[labels[i]])
        plt.axis("off")
    plt.show()


def load_cifar10_data(data_dir):
    """
    Load the CIFAR-10 dataset from the specified directory.

    This function loads all five training batches and the test batch from the CIFAR-10 dataset.
    It returns the training data (images and labels) and test data (images and labels).

    Parameters
    ----------
    data_dir : str
        The path to the CIFAR-10 data directory.

    Returns
    -------
    tuple
        A tuple (x_train, y_train, x_test, y_test) where:
        - x_train : np.ndarray of shape (50000, 3072) containing training images.
        - y_train : np.ndarray of shape (50000,) containing training labels.
        - x_test  : np.ndarray of shape (10000, 3072) containing test images.
        - y_test  : np.ndarray of shape (10000,) containing test labels.
    """
    x_train_list = []
    y_train_list = []

    # Load all five training batches
    for i in range(1, 6):
        batch_name = os.path.join(data_dir, f"data_batch_{i}")
        batch = load_cifar10_batch(batch_name)
        x_train_list.append(batch["data"])
        y_train_list.append(batch["labels"])

    # Concatenate training data and labels
    x_train = np.concatenate(x_train_list)
    y_train = np.concatenate(y_train_list)

    # Load test batch
    test_batch = load_cifar10_batch(os.path.join(data_dir, "test_batch"))
    x_test = test_batch["data"]
    y_test = np.array(test_batch["labels"])

    return x_train, y_train, x_test, y_test


def display_training_class_distribution(data):
    """
    Display a histogram showing the distribution of classes in the training dataset.

    Parameters
    ----------
    data : np.ndarray
        A NumPy array of class labels for the training dataset.

    Returns
    -------
    None
        Displays a histogram illustrating how many samples belong to each class.
    """
    plt.hist(data, label="Training set", rwidth=0.8)
    plt.title("Histogram")
    plt.ylabel("Number of samples")
    plt.xlabel("classes")
    plt.legend()
    plt.show()


def display_training_validation_class_distribution(data):
    """
    Display a histogram comparing the class distributions for training and validation sets.

    Parameters
    ----------
    data : np.ndarray
        A NumPy array of class labels (combined) for the training and validation datasets.

    Returns
    -------
    None
        Displays a histogram comparing training vs. validation class distribution.
    """
    plt.hist(data, label=("Training set", "Validation set"), rwidth=0.8)
    plt.title("Histogram")
    plt.ylabel("Number of samples")
    plt.xlabel("classes")
    plt.legend()
    plt.show()


def create_train_test_datasets_mnist(data):
    """
    Create training and testing datasets from MNIST-like data.

    This function expects a dictionary-like object where keys are strings containing
    "train" or "test", and the values are arrays/lists of data samples. Each key's
    last character represents the class label for all samples in that entry.

    Parameters
    ----------
    data : dict
        A dictionary-like structure where each key maps to a set of images. Keys
        include 'trainX' or 'testX' where X represents a digit class label.

    Returns
    -------
    tuple
        (x_train, y_train, x_test, y_test) where:
        - x_train : np.ndarray of shape (N_train, ...) containing training samples.
        - y_train : np.ndarray of shape (N_train,) containing training labels.
        - x_test  : np.ndarray of shape (N_test, ...) containing testing samples.
        - y_test  : np.ndarray of shape (N_test,) containing testing labels.
    """
    x_train, y_train, x_test, y_test = [], [], [], []

    for key, data in data:
        if "train" in key:
            for x in data:
                x_train.append(x)
                num = int(key[-1])
                y_train.append(num)
        if "test" in key:
            for x in data:
                x_test.append(x)
                num = int(key[-1])
                y_test.append(num)

    x_train = np.array(x_train, dtype=np.double)
    y_train = np.array(y_train, dtype=np.double)
    x_test = np.array(x_test, dtype=np.double)
    y_test = np.array(y_test, dtype=np.double)

    return x_train, y_train, x_test, y_test

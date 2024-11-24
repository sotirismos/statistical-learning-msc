import os
import pickle

import matplotlib.pyplot as plt
import numpy as np


def unpickle_file(file: str) -> dict:
    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict


def load_cifar10_batch(file_path: str) -> dict:
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
    # TODO: Explain why we normalize the data
    # The data is converted to float32 to ensure that the division operation results in floating-point values, preserving decimal precision.
    x_normalized = x.astype("float32") / 255.0
    return x_normalized


def display_images(images, labels, class_names, num_images):
    plt.figure(figsize=(10, 10))
    for i in range(num_images):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        plt.title(class_names[labels[i]])
        plt.axis("off")
    plt.show()


def load_cifar10_data(data_dir):
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
    plt.hist(data, label="Training set", rwidth=0.8)
    plt.title("Histogram")
    plt.ylabel("Number of digits")
    plt.xlabel("Digits")
    plt.legend()
    plt.show()


def display_training_validation_class_distribution(data):
    plt.hist(data, label=("Training set", "Validation set"), rwidth=0.8)
    plt.title("Histogram")
    plt.ylabel("Number of digits")
    plt.xlabel("Digits")
    plt.legend()
    plt.show()

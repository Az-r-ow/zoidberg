from datasets import load_dataset

dataset = load_dataset("Az-r-ow/chest_xray", split="train")

from PIL import Image
from io import BytesIO

labels = dataset.features["label"].names


def format_dataset(dataset):
    pd_dataset = dataset.to_pandas()
    pd_dataset["true_label"] = pd_dataset["label"].map(lambda x: labels[x])
    pd_dataset["image"] = pd_dataset["image"].map(
        lambda i: Image.open(BytesIO(i["bytes"]))
    )
    return pd_dataset


# Converting to pandas will encode the images to bytestrings
train_data = format_dataset(dataset)
train_data.head()


def get_center_crop_coord(image, target_size=(100, 100)):
    """
    Calculates the coordinates for center cropping an image to the specified target size.

    Parameters:
        image (PIL.Image.Image): The input image to be center cropped.
        target_size (tuple): A tuple specifying the target size (width, height) for the cropped region.

    Returns:
        tuple: A tuple containing the coordinates (x, y, width, height) for center cropping the image.
    """
    width, height = image.size
    crop_x = (width - target_size[0]) // 2
    crop_y = (height - target_size[1]) // 2
    return (crop_x, crop_y, crop_x + target_size[0], crop_y + target_size[1])


import pandas as pd


def undersample(df, true_label, frac):
    """
    Undersamples a DataFrame by reducing the number of samples for a specific class.

    Parameters:
    -----------
    df : pandas DataFrame
        The DataFrame containing the dataset to be undersampled.

    true_label : str
        The true label of the class for which to perform undersampling.

    rate : float or int
        The desired ratio of samples to retain for the specified class after undersampling.
        If float, it represents the fraction of samples to retain (0.0 to 1.0).
        If int, it represents the absolute number of samples to retain.

    Returns:
    --------
    pandas DataFrame
        The undersampled DataFrame with reduced number of samples for the specified class.

    Example:
    --------
    undersampled_df = undersample(df, 'class', 0.5)
    """
    undersampled_df = pd.concat(
        [
            df[df["true_label"] != true_label],
            df[df["true_label"] == true_label].sample(frac=frac),
        ]
    )
    return undersampled_df.reset_index(drop=True)


train_data = undersample(train_data, "PNEUMONIA", 0.8)  # Retaining 80% of the samples

import math


def oversample(df, true_label, frac):
    """
    Oversamples a DataFrame by randomly duplicating the number of samples for a specific class.

    Parameters:
    -----------
    df : pandas DataFrame
        The DataFrame containing the dataset to be oversampled.

    true_label : str
        The true label of the class for which to perform oversampling.

    rate : float or int
        The desired ratio of samples to duplicated from the specified class.
        If float, it represents the fraction of samples to duplicate (0.0 to 1.0).
        If int, it represents duplicating the entirety of the samples.

    Returns:
    --------
    pandas DataFrame
        The oversampled DataFrame with the randomly duplicated number of samples for the specified class.

    Example:
    --------
    oversampled_df = oversample(df, 'class', 0.2)
    """
    majority_class = df[df["true_label"] != true_label]
    minority_class = df[df["true_label"] == true_label]
    frac = 1 + frac if frac < 1 else frac
    num_samples = math.ceil(len(minority_class) * frac)
    oversampled_minority = minority_class.sample(
        num_samples, replace=True, random_state=42
    )
    return pd.concat([majority_class, oversampled_minority]).reset_index(drop=True)


train_data = oversample(train_data, "NORMAL", 0.2)

initial_data_num_rows = len(dataset)
balanced_train_data_num_rows = len(train_data)
avg = (initial_data_num_rows + balanced_train_data_num_rows) / 2

import numpy as np

y_train = train_data["label"].reset_index(drop=True).to_numpy()

from scipy.ndimage import center_of_mass

x_train_pca = np.load("./datasets/x_train_pca.npy")


# Fetching the test data from hugging face
test_dataset = load_dataset("Az-r-ow/chest_xray", split="test")
test_dataset = format_dataset(test_dataset)

# Transforming the images to vectors with values between [0, 1]
y_test = test_dataset["label"].to_numpy()

x_train_pca = np.load("./datasets/x_train_pca.npy")
x_test_pca = np.load("./datasets/x_test_pca.npy")

from utils.NeuralNetPy import models

network = models.Network()

from utils.NeuralNetPy import ACTIVATION, WEIGHT_INIT, layers

features_size = len(x_train_pca[0])
network.addLayer(layers.Dense(features_size))
network.addLayer(layers.Dense(128, ACTIVATION.RELU, WEIGHT_INIT.HE))
network.addLayer(layers.Dense(32, ACTIVATION.SIGMOID, WEIGHT_INIT.HE))
network.addLayer(layers.Dense(2, ACTIVATION.SOFTMAX, WEIGHT_INIT.GLOROT))

from utils.NeuralNetPy import optimizers, LOSS

# Setting up the model for training
network.setup(optimizer=optimizers.Adam(0.01), loss=LOSS.BCE)

from utils.NeuralNetPy import TrainingData2dI

# Since already normalized just pass the inputs to batch with TrainData2dI
train_data = TrainingData2dI(x_train_pca, y_train)
train_data.batch(32)

from utils.NeuralNetPy import callbacks

callbacks = [callbacks.ModelCheckpoint("checkpoints", saveBestOnly=True, verbose=False)]
train_score = network.train(train_data, 50, callbacks=callbacks, progBar=True)

predictions = network.predict(x_test_pca)

predictions = np.argmax(predictions, axis=1)

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

roc_auc = roc_auc_score(y_test, predictions)
accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print("Accuracy:", accuracy)
print("F1-score:", f1)
print("roc auc::", roc_auc)

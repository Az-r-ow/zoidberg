from PIL import Image
from io import BytesIO
from .NeuralNetPy import TrainingData2dI, callbacks
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    roc_auc_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
import csv
import os
import cv2


def format_dataset(dataset, labels):
    pd_dataset = dataset.to_pandas()
    pd_dataset["true_label"] = pd_dataset["label"].map(lambda x: labels[x])
    pd_dataset["image"] = pd_dataset["image"].map(
        lambda i: Image.open(BytesIO(i["bytes"]))
    )
    return pd_dataset


def resize_grayscale(img, size=(500, 500)):
    grayscale = img.convert("L")
    resized_img = grayscale.resize(size)
    return resized_img


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


# Displaying the images
def display_images(
    data, rows, columns, crop_area=None, scatter_coordinates=([], []), title=""
):
    if len(data) < (rows * columns):
        raise ValueError(
            f"Length data should be > than (rows * columns) we got : {rows * columns} and data length = {len(data)}"
        )

    x_scatter, y_scatter = scatter_coordinates
    is_dataframe = isinstance(data, pd.DataFrame)

    if rows <= 0 or columns <= 0:
        raise ValueError(
            f"Rows and columns can't be <= 0, got : rows = {rows} and columns = {columns}"
        )
    elif rows == 1 and columns == 1:
        fig, ax = plt.subplots()
        image = data[0] if not is_dataframe else data["image"][0]
        title = "" if not is_dataframe else data["true_label"][0]
        ax.imshow(image, cmap="gray")
        ax.set_title(title)
        plt.show()
        return

    fig, axes = plt.subplots(rows, columns, figsize=(20, 10))
    for i, ax in enumerate(axes.flat):
        image = data[i] if not is_dataframe else data["image"][i]
        label = "" if not is_dataframe else data["true_label"][i]
        ax.imshow(image, cmap="gray")
        if i < len(x_scatter) and i < len(y_scatter):
            ax.scatter(x_scatter[i], y_scatter[i])
        ax.set_title(label)

        if crop_area:
            x, y, width, height = crop_area = get_center_crop_coord(image, crop_area)
            rect = patches.Rectangle(
                (x, y),
                crop_area[0],
                crop_area[1],
                linewidth=2,
                edgecolor="r",
                facecolor="none",
            )
            ax.add_patch(rect)

        ax.axis("off")

    plt.show()


def generate_mask(image_shape):
    """
    This function is the result of the analysis done in : image_mask.ipynb
    """
    height, width = image_shape[:2]

    # Fixed offsets
    sideoffset = 7
    topoffset = 6

    # Define the points for an image of size (100, 100)
    original_points = np.array(
        [
            [32, topoffset],
            [11, 28],
            [sideoffset, 100],
            [100 - sideoffset, 100],
            [100 - 11, 28],
            [67, topoffset],
        ],
        np.int32,
    )

    # Scale the points to the new image size
    scale_x = width / 100
    scale_y = height / 100
    scaled_points = original_points * [scale_x, scale_y]

    # Reshape the points for cv2.fillPoly
    scaled_points = scaled_points.reshape((-1, 1, 2))

    # Create a mask
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [scaled_points.astype(np.int32)], color=1)

    return mask


def plot_confusion_matrix(cm, model_name):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix of {model_name}")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()


def generate_values_around_median(median, num_elements):
    # Calculate the half-size of the sequence
    half_size = num_elements // 2

    # Generate the sequence around the median
    sequence = np.arange(median - half_size, median + half_size + 1)

    return sequence


def plot_models_scores(model_scores):
    average_model_scores = {
        model: {
            metric: (np.mean(scores) if isinstance(scores, np.ndarray) else scores)
            for metric, scores in metrics.items()
        }
        for model, metrics in model_scores.items()
    }

    metrics = list(next(iter(average_model_scores.values())).keys())

    # Get the number of models
    num_models = len(average_model_scores)
    num_metrics = len(metrics)

    # Generate a list of colors
    colors = plt.cm.tab10(np.linspace(0, 1, num_models))

    fig, ax = plt.subplots()
    index = np.arange(num_metrics) * num_models * 1.5

    for j, metric in enumerate(metrics):
        medians = generate_values_around_median(index[j], num_models)
        for i, (model, scores) in enumerate(average_model_scores.items()):
            ax.bar(medians[i], scores[metric], color=colors[i])
            ax.text(
                medians[i],
                scores[metric] + 0.01,
                f"{scores[metric]:.3f}",
                ha="center",
                va="bottom",
                rotation=-45,
            )

    ax.set_xticks(index)
    ax.set_xticklabels(metrics)
    plt.legend(list(average_model_scores.keys()), loc="lower center")
    plt.show()


def train_evaluate_model(
    model,
    model_name="NeuralNetPy",
    epochs=10,
    callbacksList=[],
    stratified=False,
    shuffle=False,
    dropLast=False,
    verbose=False,
):
    # Load data
    preprocessed_data = np.load("./datasets/processed_data.npz", allow_pickle=True)
    y_train = preprocessed_data["y_train"]
    y_eval = preprocessed_data["y_test"]
    x_train_pca = np.load("./datasets/x_train_pca.npy")
    x_eval_pca = np.load("./datasets/x_test_pca.npy")

    # Split the data 90-10 with stratification
    x_train_pca, x_test_pca, y_train, y_test = train_test_split(
        x_train_pca, y_train, test_size=0.1, stratify=y_train, random_state=42
    )

    # Create a 2d TrainingData object
    train_data = TrainingData2dI(x_train_pca, y_train, x_test_pca, y_test)

    # Mini-batch the data
    train_data.batch(128, stratified, shuffle, dropLast, verbose)

    logs_filename = f"{model_name}.csv"
    # Add a csv logger to track the progress during training
    callbacksList.append(callbacks.CSVLogger(logs_filename))

    # Train the model
    if verbose:
        print("Training model")
    model.train(train_data, epochs, callbacksList, progBar=False)

    training_logs = pd.read_csv(logs_filename)

    # convert epochs to ints
    training_logs["EPOCH"] = training_logs["EPOCH"].astype("int32")

    y_pred = model.predict(x_eval_pca)
    y_pred = np.argmax(y_pred, axis=1)

    # Create a figure with 4 subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    # First plot will have a comparison between train and test losses over the epochs
    epochs = training_logs["EPOCH"].to_numpy()
    train_loss = training_logs["LOSS"].to_numpy()
    test_loss = training_logs["TEST_LOSS"].to_numpy()

    axs[0, 0].plot(epochs, train_loss, color="red", label="train_loss")
    axs[0, 0].plot(epochs, test_loss, color="blue", label="test_loss")
    axs[0, 0].set_xlabel("epochs")
    axs[0, 0].set_ylabel("loss")
    axs[0, 0].legend()

    # Second plot will have a comparison between train and test accuracy
    train_accuracy = training_logs["ACCURACY"].to_numpy()
    test_accuracy = training_logs["TEST_ACCURACY"].to_numpy()

    axs[0, 1].plot(epochs, train_accuracy, color="red", label="train_accuracy")
    axs[0, 1].plot(epochs, test_accuracy, color="blue", label="test_accuracy")
    axs[0, 1].set_xlabel("epochs")
    axs[0, 1].set_ylabel("accuracy")
    axs[0, 1].legend()
    # Third plot will have a confusion matrix on the eval set
    cm = confusion_matrix(y_eval, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=axs[1, 0])

    # Fourth plot will have bars for each score on the eval predictions
    eval_scores = {
        "eval_accuracy": accuracy_score(y_eval, y_pred),
        "eval_roc_auc": roc_auc_score(y_eval, y_pred),
        "eval_recall": recall_score(y_eval, y_pred),
    }

    idx = 1
    for score in eval_scores:
        axs[1, 1].bar(idx, eval_scores[score], label=score)
        axs[1, 1].text(
            idx,
            eval_scores[score] + 0.01,
            f"{eval_scores[score]:.3f}",
            ha="center",
            rotation=-45,
        )
        idx += 2
    axs[1, 1].legend(loc="lower right")
    # plt.legend()
    plt.show()

    return model

    # y_pred = model.predict(x_test_pca)
    # y_pred = np.argmax(y_pred, axis=1)
    # headers = ["model_name", "accuracy", "roc_auc", "recall"]
    # try:
    #     with open(csv_file, mode="x") as file:
    #         file.write("")
    # except FileExistsError:
    #     print(f"File {csv_file} already exists")
    # with open(csv_file, mode="r") as file:
    #     reader = csv.reader(file)
    #     first_row = next(reader, None)
    #     if not first_row or not any(first_row):
    #         with open(csv_file, mode="w") as w_file:
    #             writer = csv.writer(w_file)
    #             writer.writerow(headers)
    # with open(csv_file, mode="a") as file:
    #     print(f"Writing scores for {model_name}...")
    #     writer = csv.DictWriter(file, fieldnames=headers)
    #     writer.writerow(
    #         {
    #             "model_name": model_name,
    #             "accuracy": round(accuracy_score(y_test, y_pred), 3),
    #             "roc_auc": round(roc_auc_score(y_test, y_pred), 3),
    #             "recall": round(recall_score(y_test, y_pred), 3),
    #         }
    #     )


def filename_without_ext(filename):
    return os.path.splitext(os.path.basename(filename))[0]

from PIL import Image
from io import BytesIO
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np
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

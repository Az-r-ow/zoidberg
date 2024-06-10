from PIL import Image
from io import BytesIO


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

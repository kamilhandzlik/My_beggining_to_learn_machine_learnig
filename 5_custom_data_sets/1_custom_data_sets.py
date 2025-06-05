"""
04. PyTorch Custom Datasets
In the last notebook, notebook 03, we looked at how to build computer vision models on an in-built dataset in PyTorch (FashionMNIST).

The steps we took are similar across many different problems in machine learning.

Find a dataset, turn the dataset into numbers, build a model (or find an existing model) to find patterns in those numbers that can be used for prediction.

PyTorch has many built-in datasets used for a wide number of machine learning benchmarks, however, you'll often want to use your own custom dataset.

What is a custom dataset?
A custom dataset is a collection of data relating to a specific problem you're working on.

In essence, a custom dataset can be comprised of almost anything.

For example, if we were building a food image classification app like Nutrify, our custom dataset might be images of food.

Or if we were trying to build a model to classify whether or not a text-based review on a website was positive or negative, our custom dataset might be examples of existing customer reviews and their ratings.

Or if we were trying to build a sound classification app, our custom dataset might be sound samples alongside their sample labels.

Or if we were trying to build a recommendation system for customers purchasing things on our website, our custom dataset might be examples of products other people have bought.
"""

"""

Topic	Contents
0. Importing PyTorch and setting up device-agnostic code	Let's get PyTorch loaded and then follow best practice to setup our code to be device-agnostic.
1. Get data	We're going to be using our own custom dataset of pizza, steak and sushi images.
2. Become one with the data (data preparation)	At the beginning of any new machine learning problem, it's paramount to understand the data you're working with. Here we'll take some steps to figure out what data we have.
3. Transforming data	Often, the data you get won't be 100% ready to use with a machine learning model, here we'll look at some steps we can take to transform our images so they're ready to be used with a model.
4. Loading data with ImageFolder (option 1)	PyTorch has many in-built data loading functions for common types of data. ImageFolder is helpful if our images are in standard image classification format.
5. Loading image data with a custom Dataset	What if PyTorch didn't have an in-built function to load data with? This is where we can build our own custom subclass of torch.utils.data.Dataset.
6. Other forms of transforms (data augmentation)	Data augmentation is a common technique for expanding the diversity of your training data. Here we'll explore some of torchvision's in-built data augmentation functions.
7. Model 0: TinyVGG without data augmentation	By this stage, we'll have our data ready, let's build a model capable of fitting it. We'll also create some training and testing functions for training and evaluating our model.
8. Exploring loss curves	Loss curves are a great way to see how your model is training/improving over time. They're also a good way to see if your model is underfitting or overfitting.
9. Model 1: TinyVGG with data augmentation	By now, we've tried a model without, how about we try one with data augmentation?
10. Compare model results	Let's compare our different models' loss curves and see which performed better and discuss some options for improving performance.
11. Making a prediction on a custom image	Our model is trained to on a dataset of pizza, steak and sushi images. In this section we'll cover how to use our trained model to predict on an image outside of our existing dataset.
"""

# 0. Importing PyTorch and setting up device-agnostic code	Let's get PyTorch loaded and then follow best practice to setup our code to be device-agnostic.
import torch
from torch import nn

device = "cuda" if torch.cuda.is_available else "cpu"
print(f"project is on {device} device")

# 1. Get data	We're going to be using our own custom dataset of pizza, steak and sushi images.
import requests
import zipfile
from pathlib import Path

data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"

if image_path.is_dir():
    print(f"{image_path} directory alreasy exists.")
else:
    print(f"Did not found { image_path} directory, creating one...")
    image_path.mkdir(parents=True, exist_ok=True)

    with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
        requests = requests.get(
            "https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip"
        )
        print("Donloading pizza, steak, sushi data ...")
        f.write(requests.content)

    with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
        print("Unziping pizza, steak, sushi file...")
        zip_ref.extractall(image_path)


# 2. Become one with the data (data preparation)	At the beginning of any new machine learning problem, it's paramount to understand the data you're working with. Here we'll take some steps to figure out what data we have.
import os
import sys

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def walk_through_dir(dirpath):
    for dirpath, dirnames, filenames in os.walk(dirpath):
        print(
            f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'."
        )


print(walk_through_dir(dirpath=image_path))

# Setup train and testing paths
train_dir = image_path / "train"
test_dir = image_path / "test"

print(train_dir, test_dir)

# Visualizing image
"""
1. Get all of the image paths using pathlib.Path.glob() to find all of the files ending in .jpg.
2. Pick a random image path using Python's random.choice().
3. Get the image class name using pathlib.Path.parent.stem.
4. And since we're working with images, we'll open the random image path using PIL.Image.open() (PIL stands for Python Image Library).
5. We'll then show the image and print some metadata.
"""

import random
from PIL import Image
import matplotlib.pyplot as plt

# 1. Set seed
random.seed(42)
# 2. Get all image paths
image_path_list = list(image_path.glob("*/*/*.jpg"))
# 3. Get random image path
random_image_path = random.choice(image_path_list)
# 4. Get image class from path name
image_class = random_image_path.parent.stem
# 5. Open image
img = Image.open(random_image_path)

# 6. Print metadata
print(f"Random image path: {random_image_path}")
print(f"Image class: {image_class}")
print(f"Image height: {img.height}")
print(f"Image width: {img.width}")

# Odkomentuj żeby wyświetlić obraz
# plt.imshow(img)
# plt.axis(False)
# plt.title(image_class)
# plt.show()


# We can do the same with matplotlib.pyplot.imshow(), except we have to convert the image to a NumPy array first.
import numpy as np
import matplotlib.pyplot as plt


img_as_array = np.asarray(img)

# Odkomentuj żeby wyświetlić obraz
# plt.figure(figsize=(10, 7))
# plt.imshow(img_as_array)
# plt.title(
#     f"Image class: {image_class} | Image shape: {img_as_array.shape} -> [height, width, color_channels]"
# )
# plt.axis(False)
# plt.show()

# Odkomentuj żeby wyświetlić obraz jako zbiór danych w terminalu
# print(img_as_array)


# 3. Transforming data	Often, the data you get won't be 100% ready to use with a machine learning model, here we'll look at some steps we can take to transform our images so they're ready to be used with a model.
# 3.1 turn your data into tensors (tutaj numeryczne przedstawienie obrazu na tensor)
# 3.2 Zamień te dane na torch.utils.data DataLoader i torchvision dataset

# 3.1
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

data_transform = transforms.Compose(
    [
        # Resize images to 64x64 px
        transforms.Resize(size=(64, 64)),
        # Flip the images randomly
        transforms.RandomHorizontalFlip(p=0.5),
        # Turn Image to torch.Tensor
        transforms.ToTensor(),
    ]
)


def plot_transformed_images(image_paths, transform, n=3, seed=42):
    """Plots a series of random images from image_paths.

    Will open n image paths from image_paths, transform them
    with transform and plot them side by side.

    Args:
        image_paths (list): List of target image paths.
        transform (PyTorch Transforms): Transforms to apply to images.
        n (int, optional): Number of images to plot. Defaults to 3.
        seed (int, optional): Random seed for the random generator. Defaults to 42.
    """
    random.seed(seed)
    random_image_paths = random.sample(image_paths, k=n)
    for image in random_image_paths:
        with Image.open(image) as f:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(f)
            ax[0].set_title(f"Original \nSize: {f.size}")
            ax[0].axis("off")

            # Transform and plot image
            # Note: permute() will change shape of image to suit matplotlib
            # (PyTorch default is [C, H, W] but Matplotlib is [H, W, C])
            transformed_image = transform(f).permute(1, 2, 0)
            ax[1].imshow(transformed_image)
            ax[1].set_title(f"Transformed \nSize: {transformed_image.shape}")
            ax[1].axis("off")

            fig.suptitle(f"Class: {image_path.parent.stem}", fontsize=16)
    plt.show()


# plot_transformed_images(image_paths=image_path_list, transform=data_transform, n=3)


# Use ImageFolder to create dataset(s)
from torchvision import datasets

train_data = datasets.ImageFolder(
    root=train_dir,  # target folder of images
    transform=data_transform,  # transforms to perform on data (images)
    target_transform=None,
)  # transforms to perform on labels (if necessary)

test_data = datasets.ImageFolder(root=test_dir, transform=data_transform)

print(f"Train data:\n{train_data}\nTest data:\n{test_data}")


# Get class names as list
class_names = train_data.classes
# Same but turning names into dict
class_dict = train_data.class_to_idx
# Check the lengths
print(f"Train data: {len(train_data)} | Tet data: {len(test_data)}")

img, label = train_data[0][0], train_data[0][1]
print(f"Image tensor:\n{img}")
print(f"Image shpae: {img.shape}")
print(f"Image datatype: {img.dtype}")
print(f"Image label: {label}")
print(f"Label datatype: {type(label)}")

img_permute = img.permute(1, 2, 0)

# Print out different shapes (before and after permute)
print(f"Original shape: {img.shape} -> [color_channels, height, width]")
print(f"Image permute shape: {img_permute.shape} -> [height, width, color_channels]")

# Plot image
# plt.figure(figsize=(10, 7))
# plt.imshow(img.permute(1, 2, 0))
# plt.axis("off")
# plt.title(class_names[label], fontsize=14)
# plt.show()


# 4.1 Turn loaded images into DataLoader's
BATCH_SIZE = 32
train_dataloader = DataLoader(
    dataset=train_data, batch_size=BATCH_SIZE, num_workers=0, shuffle=True
)
test_dataloader = DataLoader(
    dataset=test_data, batch_size=BATCH_SIZE, num_workers=0, shuffle=True
)

print(
    f"Train data loader:\n{train_dataloader}\n-------------------\nTest data loader:\n{test_dataloader}"
)
print(
    f"lenght of train data loader: {len(train_dataloader)} | Lenght of test dataloader {len(test_dataloader)}"
)
# Batch size will now be 1, try changing the batch_size parameter above and see what happens
img, label = next(iter(train_dataloader))
print(f"Image shape: {img.shape} -> [batch_size, color_channels, height, width]")
print(f"Label shape: {label.shape}")


#  Option 2: Loading Image Data with a Custom Dataset
# ------------------------------------------------------
import os
import pathlib
import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Tuple, Dict, List

# Instance of torchvision.datasets.ImageFolder()
train_data.classes, train_data.class_to_idx

# Setup path for target directory
target_directory = train_dir
print(f"Target directory: {target_directory}")

# Get the class names from the target directory
class_names_found = sorted(
    [entry.name for entry in list(os.scandir(image_path / "train"))]
)
print(f"Class names found: {class_names_found}")


# Make function to find classes in target directory
def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folder names in a target directory.

    Assumes target directory is in standard image classification format.

    Args:
        directory (str): target directory to load classnames from.

    Returns:
        Tuple[List[str], Dict[str, int]]: (list_of_class_names, dict(class_name: idx...))

    Example:
        find_classes("food_images/train")
        >>> (["class_1", "class_2"], {"class_1": 0, ...})
    """
    # 1. Get the class names by scanning the target directory
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())

    # 2. Raise an error if class names not found
    if not classes:
        raise FileNotFoundError(f"Couldn't find any classes in {directory}.")

    # 3. Create a dictionary of index labels (computers prefer numerical rather than string labels)
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


# print(find_classes(train_dir))

# 5.2 Create a custom Dataset to replicate ImageFolder
# Write a custom dataset class (inherits from torch.utils.data.Dataset)


class ImageFolderCustom(Dataset):
    def __init__(self, targ_dir: str, transform=None) -> None:
        # 3. Create class attributes
        # Get all image paths
        self.paths = list(pathlib.Path(targ_dir).glob("*/*,jpg"))
        # Setup transforms
        self.transform = transform
        # Create classes and class_to_idx attributes
        self.classess, self.class_to_idx = find_classes(targ_dir)

    def load_image(self, index: int) -> Image.Image:
        "Opens an image via a path and returns it."
        image_path = self.paths[index]
        return Image.open(image_path)

    def __len__(self) -> int:
        "Returns the total number os samples."
        return len(self.paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        "Returns one sample of data, data and label (X, y)."
        img = self.load_image(index)
        class_name = self.paths[index].parent.name
        class_idx = self.class_to_idx[class_name]

        if self.transform:
            return self.transform(img), class_idx
        else:
            return img, class_idx


# Augment train data
train_transforms = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
    ]
)

# Don't augment test data, only reshape
test_transforms = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor])


train_data_custom = ImageFolderCustom(targ_dir=train_dir, transform=train_transforms)

test_data_custom = ImageFolderCustom(targ_dir=test_dir, transform=test_transforms)

# print(train_data_custom, test_data_custom)

# len(train_data_custom), len(test_data_custom)

# print(train_data_custom.classes)

# print(train_data_custom.class_to_idx)

# Check for equality amongst our custom Dataset and ImageFolder Dataset
# print(
# (len(train_data_custom) == len(train_data))
# & (len(test_data_custom) == len(test_data))
# )
# print(train_data_custom.classes == train_data.classes)
# print(train_data_custom.class_to_idx == train_data.class_to_idx)


# 5.3 Create a function to display random images
# 1. Take in a Dataset as well as a list of class names
def display_random_images(
    dataset: torch.utils.data.dataset.Dataset,
    classes: List[str] = None,
    n: int = 10,
    display_shape: bool = True,
    seed: int = None,
):

    # 2. Adjust display if n too high
    if n > 10:
        n = 10
        display_shape = False
        print(
            f"For display purposes, n shouldn't be larger than 10, setting to 10 and removing shape display."
        )

    # 3. Set random seed
    if seed:
        random.seed(seed)

    # 4. Get random sample indexes
    random_samples_idx = random.sample(range(len(dataset)), k=n)

    # 5. Setup plot
    plt.figure(figsize=(16, 8))

    # 6. Loop through samples and display random samples
    for i, targ_sample in enumerate(random_samples_idx):
        targ_image, targ_label = dataset[targ_sample][0], dataset[targ_sample][1]

        # 7. Adjust image tensor shape for plotting: [color_channels, height, width] -> [color_channels, height, width]
        targ_image_adjust = targ_image.permute(1, 2, 0)

        # Plot adjusted samples
        plt.subplot(1, n, i + 1)
        plt.imshow(targ_image_adjust)
        plt.axis("off")
        if classes:
            title = f"class: {classes[targ_label]}"
            if display_shape:
                title = title + f"\nshape: {targ_image_adjust.shape}"
        plt.title(title)
    plt.show()


display_random_images(train_data, n=5, classes=class_names, seed=None)

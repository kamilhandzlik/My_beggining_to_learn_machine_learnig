#################################################################################
# Część dalsza praktycznie to samo ale z użyciem convolutional neural network ###
#################################################################################

# Ważne!!! Przeczytaj
# https://poloclub.github.io/cnn-explainer/#article-convolution

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from timeit import default_timer
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from helper_function import accuracy_fn


# 1. Przygotowanie danych
train_data = datasets.FashionMNIST(
    root="data",
    train=True,
    transform=ToTensor(),
    target_transform=None,
)
test_data = datasets.FashionMNIST(
    root="data",
    train=True,
    transform=ToTensor(),
    target_transform=None,
)

RANDOM_SEED = 42
BATCH_SIZE = 32
device = "cuda" if torch.cuda.is_available() else "gpu"
print(f"Works on {device}...")

train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

class_names = train_data.classes
class_to_idx = train_data.class_to_idx

train_features_batch, train_labels_batch = next(iter(train_dataloader))
print(train_features_batch.shape, train_labels_batch.shape)

flatten_model = nn.Flatten()
x = train_features_batch[0]
output = flatten_model(x)


class FashinMNISTModelV2(nn.Module):
    """
    Model, któy odtwarza architekturę TinyVGG Convolutionsal Neural Network
    """

    def __init__(self, input_shape: int, hiddens_units: int, output_shape: int):
        super().__init__()
        self.conv_block_1 = (
            nn.Conv2d(
                in_channels=input_shape,
                out_channels=hiddens_units,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hiddens_units,
                out_channels=hiddens_units,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv_block_2 = (
            nn.Conv2d(
                in_channels=hiddens_units,
                out_channels=hiddens_units,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hiddens_units,
                out_channels=hiddens_units,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hiddens_units, out_features=output_shape),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x


torch.manual_seed(RANDOM_SEED)
model_2 = FashinMNISTModelV2(
    input_shape=1, hiddens_units=10, output_shape=len(class_names)
).to(device)


# 3. Pętle trenigowa i testowa

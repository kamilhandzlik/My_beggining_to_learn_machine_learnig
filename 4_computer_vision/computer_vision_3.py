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
        self.conv_block_1 = nn.Sequential(
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
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=hiddens_units,
                out_channels=hiddens_units,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hiddens_units,
                out_channels=hiddens_units,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hiddens_units * 7 * 7, out_features=output_shape),
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

torch.manual_seed(42)

# Create sample batch of random numbers with same size as image batch
# images = torch.randn(
# size=(32, 3, 64, 64)
# )  # [batch_size, color_channels, height, width]
# test_image = images[0]  # get a single image for testing
# print(
# f"Image batch shape: {images.shape} -> [batch_size, color_channels, height, width]"
# )
# print(f"Single image shape: {test_image.shape} -> [color_channels, height, width]")
# print(f"Single image pixel values:\n{test_image}")


torch.manual_seed(42)

# Create a convolutional layer with same dimensions as TinyVGG
# (try changing any of the parameters and see what happens)
# conv_layer = nn.Conv2d(
# in_channels=3, out_channels=10, kernel_size=3, stride=1, padding=0
# )  # also try using "valid" or "same" here

# Pass the data through the convolutional layer
# conv_layer(
# test_image
# )  # Note: If running PyTorch <1.11.0, this will error because of shape issues (nn.Conv.2d() expects a 4d tensor as input)
# Check out the conv_layer_2 internal parameters
# print(conv_layer_2.state_dict())
# Get shapes of weight and bias tensors within conv_layer_2
# print(f"conv_layer_2 weight shape: \n{conv_layer_2.weight.shape} -> [out_channels=10, in_channels=3, kernel_size=5, kernel_size=5]")
# print(f"\nconv_layer_2 bias shape: \n{conv_layer_2.bias.shape} -> [out_channels=10]")


# Print out original image shape without and with unsqueezed dimension
# print(f"Test image original shape: {test_image.shape}")
# print(f"Test image with unsqueezed dimension: {test_image.unsqueeze(dim=0).shape}")

# Create a sample nn.MaxPoo2d() layer
# max_pool_layer = nn.MaxPool2d(kernel_size=2)

# Pass data through just the conv_layer
# test_image_through_conv = conv_layer(test_image.unsqueeze(dim=0))
# print(f"Shape after going through conv_layer(): {test_image_through_conv.shape}")

# Pass data through the max pool layer
# test_image_through_conv_and_max_pool = max_pool_layer(test_image_through_conv)
# print(f"Shape after going through conv_layer() and max_pool_layer(): {test_image_through_conv_and_max_pool.shape}")

# torch.manual_seed(42)
# Create a random tensor with a similar number of dimensions to our images
# random_tensor = torch.randn(size=(1, 1, 2, 2))
# print(f"Random tensor:\n{random_tensor}")
# print(f"Random tenso/r shape: {random_tensor.shape}")

# Create a max pool layer
# max_pool_layer = nn.MaxPool2d(kernel_size=2) # see what happens when you change the kernel_size value

# Pass the random tensor through the max pool layer
# max_pool_tensor = max_pool_layer(random_tensor)
# print(f"\nMax pool tensor:\n{max_pool_tensor} <- this is the maximum value from random_tensor")
# print(f"Max pool tensor shape: {max_pool_tensor.shape}")


# 3. Pętle trenigowa i testowa
epochs = 3
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_2.parameters(), lr=0.1)


def print_train_time(start: float, end: float, device: torch.device = None):
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time


start_time = default_timer()
end_time = default_timer()
print(print_train_time(start=start_time, end=end_time, device=device))

torch.manual_seed(RANDOM_SEED)
train_time_star_on_gpu = default_timer()
train_time_star_model_2 = default_timer()


def train_step(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    accuracy_fn,
    device: torch.device = device,
):
    train_loss, train_acc = 0, 0
    model.to(device)
    for batch, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)
        model_2.train()
        # 1. Forward pass
        y_pred = model_2(X)
        # 2. Calculat the loss
        loss = loss_fn(y_pred, y)
        train_loss += loss
        # 3. optimizer zero grad
        optimizer.zero_grad()
        # 4. backward
        loss.backward()
        # 5. Optimizer step
        optimizer.step()
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss} | Train accuracy: {train_acc}%")


def test_step(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    accuracy_fn,
    device: torch.device = device,
):
    test_loss, test_acc = 0, 0
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            # 1. Forward pass
            test_pred = model(X)
            # 2. Calcualte loss and accuracy
            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(
                y_true=y,
                y_pred=test_pred.argmax(dim=1),
            )
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")


for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n------------")
    train_step(
        model=model_2,
        data_loader=train_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        accuracy_fn=accuracy_fn,
        device=device,
    )
    test_step(
        model=model_2,
        data_loader=train_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        accuracy_fn=accuracy_fn,
        device=device,
    )

train_time_end_model_2 = default_timer()
totalo_train_time_model_2 = print_train_time(
    start=train_time_star_model_2, end=train_time_end_model_2, device=device
)

# No to do tego momentu był przykład jak to wszystko działa na cpu czas przejść na gpu

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

# 1 Przygotowanie danych
train_data = datasets.FashionMNIST(
    root="data",
    train=True,
    transform=ToTensor(),
    target_transform=None,
)
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
    target_transform=None,
)

RANDOM_SEED = 42
BATCH_SIZE = 32
device = "cuda" if torch.cuda.is_available() else "gpu"
# print(f"len Train data: {len(train_data)}, len test data: {len(test_data)}")

image, label = train_data[0]
class_names = train_data.classes
class_to_idx = train_data.class_to_idx

train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

# Sprawdzenie co jest w środku dataloaderó:
train_features_batch, train_labels_batch = next(iter(train_dataloader))
print(train_features_batch.shape, train_labels_batch.shape)

## odkomentuj żeby zobaczyć obrazek
# plt.imshow(image.squeeze(), cmap="gray")
# plt.title(class_names[label])
# plt.axis(False)
# plt.show()

## odkomentuj żeby zobaczyć obrazek
# torch.manual_seed(RANDOM_SEED)
# fig = plt.figure(figsize=(9, 9))
# rows, cols = 4, 4
# for i in range(1, rows * cols + 1):
#     random_idx = torch.randint(0, len(train_data), size=[1]).item()
#     img, label = train_data[random_idx]
#     fig.add_subplot(rows, cols, i)
#     plt.imshow(img.squeeze(), cmap="gray")
#     plt.title(class_names[label])
#     plt.axis(False)
# plt.show()

## odkomentuj żeby zobaczyć obrazek
# Show Samples
# torch.manual_seed(RANDOM_SEED)
# random_idx = torch.randint(0, len(train_features_batch), size=[1]).item()
# img, labels = train_features_batch[random_idx], train_labels_batch[random_idx]
# plt.imshow(img.squeeze(), cmap="gray")
# plt.title(class_names[label])
# plt.axis(False)
# plt.show()
# print(f"Image size: {img.shape}")
# print(f"Label: {label}, label size: {label.shape}")

# 2 Budowanie modelu
flatten_model = nn.Flatten()
x = train_features_batch[0]
output = flatten_model(x)


#  Model
class FashionMNISTModelV1(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, ouput_shappe: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=ouput_shappe),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_stack(x)


torch.manual_seed(RANDOM_SEED)
model_1 = FashionMNISTModelV1(input_shape=784, hidden_units=10, ouput_shappe=10).to(
    device
)

#  3. Petle trenigowa i testowa
epochs = 3
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_1.parameters(), lr=0.1)


def print_train_time(start: float, end: float, device: torch.device = None):
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time


start_time = default_timer()
end_time = default_timer()
print(print_train_time(start=start_time, end=end_time, device=device))

torch.manual_seed(RANDOM_SEED)
train_time_star_on_cpu = default_timer()


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
        # 1. Forward pass
        y_pred = model_1(X)
        # 2. Calculate the loss
        loss = loss_fn(y_pred, y)
        train_loss += loss
        # 3 Optimizer zero grad
        optimizer.zero_grad()
        # 4. Backward
        loss.backward()
        # 5. optimizer step
        optimizer.step()

    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")


def test_step(
    data_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    accuracy_fn,
    device: torch.device = device,
):
    test_loss, test_acc = 0, 0
    model.to(device)
    model.eval()  # put model in eval mode
    # Turn on inference context manager
    with torch.inference_mode():
        for X, y in data_loader:
            # Send data to GPU
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred = model(X)

            # 2. Calculate loss and accuracy
            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(
                y_true=y,
                y_pred=test_pred.argmax(dim=1),  # Go from logits -> pred labels
            )

        # Adjust metrics and print out
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")


for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n------------")

    # Training loop
    train_step(
        data_loader=train_dataloader,
        model=model_1,
        loss_fn=loss_fn,
        optimizer=optimizer,
        accuracy_fn=accuracy_fn,
    )
    # Testing loop
    test_step(
        data_loader=test_dataloader,
        model=model_1,
        loss_fn=loss_fn,
        accuracy_fn=accuracy_fn,
    )


train_time_end_on_gpu = default_timer()
total_train_time_model_0 = print_train_time(
    start=train_time_star_on_cpu,
    end=train_time_end_on_gpu,
    device=str(next(model_1.parameters()).device),
)

#  4. Make predictions and get Model 0 results
torch.manual_seed(RANDOM_SEED)


def eval_model(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    accuracy_fn,
):
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in tqdm(data_loader):
            y_pred = model(X.to(device))
            loss += loss_fn(y_pred, y.to(device))
            acc += accuracy_fn(y_true=y.to(device), y_pred=y_pred.argmax(dim=1))

        loss /= len(data_loader)
        acc /= len(data_loader)

    return {
        "model_name": model.__class__.__name__,
        "model_loss": loss.item(),
        "model_acc": acc,
    }


model_1_results = eval_model(
    model=model_1, data_loader=test_dataloader, loss_fn=loss_fn, accuracy_fn=accuracy_fn
)
print(model_1_results)

#######################################
###########  Ważne!!!! ################
#######################################

"""
Note: The training time on CUDA vs CPU will depend largely on the quality of the CPU/GPU you're using. Read on for a more explained answer.

Question: "I used a GPU but my model didn't train faster, why might that be?"

Answer: Well, one reason could be because your dataset and model are both so small (like the dataset and model we're working with) the benefits of using a GPU are outweighed by the time it actually takes to transfer the data there.

There's a small bottleneck between copying data from the CPU memory (default) to the GPU memory.

So for smaller models and datasets, the CPU might actually be the optimal place to compute on.

But for larger datasets and models, the speed of computing the GPU can offer usually far outweighs the cost of getting the data there.

However, this is largely dependent on the hardware you're using. With practice, you will get used to where the best place to train your models is.
"""

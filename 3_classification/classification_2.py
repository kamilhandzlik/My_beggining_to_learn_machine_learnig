"""
Let;s imporeve model from classification_1 by:
* Adding more hidden layers
* Increase the numer of layers: 2 -> 3
* Increas the number of epochs 100 -> 1000
"""

# importy
import torch
from torch import nn
import sklearn
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import requests
from pathlib import Path
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from helper_function import plot_predictions, plot_decision_boundary


# 1. Make classification data and get it ready
#  Make 1000 samples
n_samples = 1000

# Create circles
X, y = make_circles(n_samples, noise=0.03, random_state=42)


# Make DataFrame of circle data
circles = pd.DataFrame({"X1": X[:, 0], "X2": X[:, 1], "label": y})
# print(f"First 5 samples of circles:\n{circles.head(10)}")


# Visualize the data
def plot_circles(x=X, y=y):
    plt.scatter(x=X[:, 0], y=X[:, 1], c=y, cmap=plt.cm.RdYlBu)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title("Circles dataset")
    plt.show()


# odkomentuj, aby zobaczyć wykres
# plot_circles()

# Turn data into tensors
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

# Podzielenie danych na zbiót treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 2. Create a model
# 1.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# 2.
# 2.1
class CircleModelV1(nn.Module):
    def __init__(self):
        super().__init__()
        ## 2.2 Increase number of layers
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # z = self.layer_1(x)
        # z = self.layer_2(z)
        # z = self.layer_3(z) one of the ways u can implement said layers but this is silghtly slwer
        return self.layer_3(self.layer_2(self.layer_1(x)))


# 2.4
model_1 = CircleModelV1().to(device)

loss_fn = nn.BCEWithLogitsLoss()  # Binary Cross Entropy Loss with logits
optimizer = torch.optim.SGD(params=model_1.parameters(), lr=0.1)


# Calculate accuracy
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


# 4. Training and testing loop
torch.manual_seed(42)
epochs = 1000

X_train = X_train.to(device)
X_test = X_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)

y_logits = model_1(X_test.to(device))[:5]
y_pred_probs = torch.sigmoid(y_logits)
y_preds = torch.round(y_pred_probs)
y_pred_labels = torch.round(torch.sigmoid(model_1(X_test.to(device))[:5]))

## Get rid of extra dimension
y_preds.squeeze()

for epoch in range(epochs):
    model_1.train()

    # 4.1 Forward pass
    y_logits = model_1(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))

    # 4.2 Calculating loss
    loss = loss_fn(y_logits, y_train)
    acc = accuracy_fn(y_true=y_train, y_pred=y_pred)

    # 4.3 Optimizer zero grad
    optimizer.zero_grad()

    # 4.4 Backward pass
    loss.backward()

    # 4.5 Optimizer step
    optimizer.step()

    # Testig
    model_1.eval()
    with torch.inference_mode():
        # 1. Forward pass
        test_logits = model_1(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))

        # 2. Calculate loss and accuracy
        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)

    if epoch % 100 == 0:
        print(
            f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%"
        )

""" Odkomentuj żeby zobaczyć wykres"""
# Plot decision boundary of the model
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.title("Train")
# plot_decision_boundary(model_1, X_train, y_train)
# plt.subplot(1, 2, 1)
# plt.title("Test")
# plot_decision_boundary(model_1, X_test, y_test)
# plt.show()

# Model still does not work properly testing on diffrent data were problem may lay

# Preparing data to see if our model can fit a straight line
weight = 0.7
bias = 0.3
start = 0
end = 1
step = 0.01

# Create Data
X_regression = torch.arange(start, end, step).unsqueeze(dim=1)
y_regression = weight * X_regression + bias

print(len(X_regression))
print(X_regression[:5], y_regression[:5])


train_split = int(0.8 * len(X_regression))
X_train_regression, y_train_regression = (
    X_regression[:train_split],
    X_regression[:train_split],
)
X_test_regression, y_test_regressio = (
    X_regression[train_split:],
    X_regression[train_split:],
)


# print(f"len x: {len(X_train_regression)}, {len(y_train_regression)}, {len(X_test_regression)} {len(y_test_regressio)}")
def plot_predictions(
    train_data=X_train,
    train_labels=y_train,
    test_data=X_test,
    test_labels=y_test,
    predictions=None,
):
    plt.figure(figsize=(10, 7))
    plt.scatter(train_data, train_labels, c="b", s=4, label="Train data")
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    plt.legend(prop={"size": 14})
    plt.title("Train and test data", fontsize=20)
    plt.show()


# plot_predictions(train_data=X_train_regression,
#  train_labels=y_train_regression,
#  test_data=X_test_regression,
#  test_labels=y_test_regressio)

# adjusting model to fit a straigth line
model_2 = nn.Sequential(
    nn.Linear(in_features=1, out_features=10),
    nn.Linear(in_features=10, out_features=10),
    nn.Linear(in_features=10, out_features=1),
).to(device)

loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model_2.parameters(), lr=0.1)

# Train model
torch.manual_seed(42)
torch.cuda.manual_seed(42)
epochs = 1000

X_train_regression, y_train_regression = X_train_regression.to(
    device
), y_train_regression.to(device)
X_test_regression, y_test_regressio = X_test_regression.to(device), y_test_regressio.to(
    device
)

for epoch in range(epochs):
    y_pred = model_2(X_train_regression)
    loss = loss_fn(y_pred, y_train_regression)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model_2.eval
    with torch.inference_mode():
        test_pred = model_2(X_test_regression)
        test_loss = loss_fn(test_pred, y_test_regressio)
    if epoch % 100 == 0:
        print(f"epoch {epoch}, | Loss: {loss} | Test loss: {test_loss}")


with torch.inference_mode():
    y_preds = model_2(X_test_regression)

plot_predictions(
    train_data=X_train_regression.cpu().numpy(),
    train_labels=y_train_regression.cpu().numpy(),
    test_data=X_test_regression.cpu().numpy(),
    test_labels=y_test_regressio.cpu().numpy(),
    predictions=y_preds.cpu().numpy(),
)


##################################################
############## Non linearity  ####################
##################################################

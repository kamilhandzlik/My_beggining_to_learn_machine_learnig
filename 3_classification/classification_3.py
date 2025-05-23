##################################################
############## Non linearity  ####################
##################################################
"""
6. Missing piece in previous chapter was non linaerity below I will show how to properly make
model that predicts non linear data
"""
# 6.1
# Imports
import torch
from torch import nn
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from helper_function import plot_predictions, plot_decision_boundary

device = "cuda" if torch.cuda.is_available() else "spu"
print(f"device: {device}")
# Recreating non linear data
n_samples = 1000
X, y = make_circles(n_samples, noise=0.03, random_state=42)
circles = pd.DataFrame({"X1": X[:, 0], "X2": X[:, 1], "label": y})


def plot_circles(x=X, y=y):
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.RdYlBu)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title("Circles dataset")
    plt.show()


# plot_circles()
# convert data to tensors
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# print(X_train[:5], y_train[:5])

# 6.2 Building model with non linearity
"""
Linear = straight license
Non linear = not straight lines curavture and other
"""


class CircleModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))


model_3 = CircleModelV2().to(device)
print(model_3)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params=model_3.parameters(), lr=0.1)


def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


epochs = 2000

X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

y_logits = model_3(X_test.to(device))[:5]
y_pred_probs = torch.sigmoid(y_logits)
y_preds = torch.round(y_pred_probs)
y_pred_labels = torch.round(torch.sigmoid(model_3(X_test.to(device))[:5]))

y_preds.squeeze()

for epoch in range(epochs):
    model_3.train()
    # 1. Forward pass
    y_logits = model_3(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))
    # 2. Calculate loss
    loss = loss_fn(y_logits, y_train)
    acc = accuracy_fn(y_true=y_train, y_pred=y_pred)
    # 3. Optimizer zero grad gradiant descent
    optimizer.zero_grad()
    # 4. Backpropagation
    loss.backward()
    # 5. Optimizer step
    optimizer.step()

    model_3.eval()
    with torch.inference_mode():
        # 1. Forward pass
        test_logits = model_3(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))

        # 2. Calculate loss and accuracy
        test_loss = loss_fn(test_logits, test_pred)
        test_acc = accuracy_fn(y_pred=y_test, y_true=test_pred)

    if epoch % 100 == 0:
        print(
            f"Epoch {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f} Test acc: {test_acc:.2f}%"
        )


model_3.eval()
with torch.inference_mode():
    y_preds = torch.round(torch.sigmoid(model_3(X_test))).squeeze()

print(y_preds[:10], y_test[:10])

""" Odkomentuj żeby zobaczyć wykres"""
## Plot decision boundary of the model
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_3, X_train, y_train)
plt.subplot(1, 2, 1)
plt.title("Test")
plot_decision_boundary(model_3, X_test, y_test)
plt.show()


"""
7. Replicating non-linear activation functions
We saw before how adding non-linear activation functions to our model can help it to model non-linear data.

Note: Much of the data you'll encounter in the wild is non-linear (or a combination of linear and non-linear). Right now we've been working with dots on a 2D plot. But imagine if you had images of plants you'd like to classify, there's a lot of different plant shapes. Or text from Wikipedia you'd like to summarize, there's lots of different ways words can be put together (linear and non-linear patterns).

But what does a non-linear activation look like?

How about we replicate some and what they do?

Let's start by creating a small amount of data.
"""

# Create a toy tensor (similar to the data going into our model(s))
A = torch.arange(-10, 10, 1, dtype=torch.float32)
print(A)
# Visualize the toy tensor
plt.plot(A)

"""
Now let's see how the ReLU activation function influences it.

And instead of using PyTorch's ReLU (torch.nn.ReLU), we'll recreate it ourselves.

The ReLU function turns all negatives to 0 and leaves the positive values as they are.
"""


def relu(x):
    return torch.maximum(torch.tensor(0), x)  # inputs must be tensors


# Pass toy tensor through ReLU function
relu(A)
plt.plot(relu(A))

"""
How about we try the sigmoid function we've been using?

The sigmoid function formula goes like so:

              1
outi = ---------------
        1 + e^(-inputi) 

Or using x as input:


              1
S(x) = -----------------
        1 + e^(-2xi)

Where 
S stands for sigmoid, 
e stands for exponential (torch.exp()) and 
i stands for a particular element in a tensor.

Let's build a function to replicate the sigmoid function with PyTorch.
"""


# Create a custom sigmoid function
def sigmoid(x):
    return 1 / (1 + torch.exp(-x))


# Test custom sigmoid on toy tensor
sigmoid(A)
# Plot sigmoid activated toy tensor
plt.plot(sigmoid(A))
